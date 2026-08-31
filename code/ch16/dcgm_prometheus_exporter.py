#!/usr/bin/env python3

"""
DCGM/Prometheus Integration for LLM Inference Monitoring

This module implements monitoring integration as described in Chapter 16, 
collecting GPU metrics through DCGM and exposing them to Prometheus.

Key metrics monitored:
- GPU utilization (DCGM_FI_DEV_GPU_UTIL)
- Memory copy engine utilization (DCGM_FI_DEV_MEM_COPY_UTIL)
- Framebuffer memory used (DCGM_FI_DEV_FB_USED)
- NVLink error counters and throughput
- GPU temperature and power
- PCIe and NVLink throughput

Usage:
    python dcgm_prometheus_exporter.py --port 8000
    
Then configure Prometheus to scrape from http://localhost:8000/metrics
"""

import argparse
import time
import socket
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from prometheus_client import start_http_server, Gauge, Counter, Histogram, Info
import subprocess
import json

try:
    import pydcgm
    DCGM_AVAILABLE = True
except ImportError:
    DCGM_AVAILABLE = False
    print("Warning: pydcgm not available. Install with: pip install dcgm-python")

if DCGM_AVAILABLE:
    try:
        import dcgm_structs
        import dcgm_fields
        import dcgm_agent
        import dcgm_field_helpers
    except Exception as dcgm_exc:  # pragma: no cover - informational path
        DCGM_AVAILABLE = False
        print(f"Warning: DCGM bindings incomplete: {dcgm_exc}")

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available. Install with: pip install nvidia-ml-py")
except pynvml.NVMLError as e:
    NVML_AVAILABLE = False
    print(f"Warning: NVML initialization failed: {e}")


@dataclass
class GPUMetrics:
    """Container for GPU metrics from DCGM and NVML"""
    gpu_id: int
    gpu_util: float  # percent
    mem_util: float  # percent
    mem_used: int  # bytes
    mem_total: int  # bytes
    temperature: float  # celsius
    power_usage: float  # watts
    pcie_tx: int  # bytes/sec
    pcie_rx: int  # bytes/sec
    nvlink_tx: int  # bytes/sec
    nvlink_rx: int  # bytes/sec
    sm_clock: int  # MHz
    memory_clock: int  # MHz
    encoder_util: float  # percent
    decoder_util: float  # percent
    nvlink_errors: int = 0  # total error count
    # Extended metrics for deep profiling
    pcie_replay_counter: int = 0  # PCIe link quality indicator
    xid_errors: int = 0  # GPU XID error events
    retired_pages_sbe: int = 0  # Single-bit ECC retired pages
    retired_pages_dbe: int = 0  # Double-bit ECC retired pages
    sm_active: float = 0.0  # SM activity percentage (profiling)
    sm_occupancy: float = 0.0  # SM occupancy percentage (profiling)
    tensor_active: float = 0.0  # Tensor core activity percentage
    dram_active: float = 0.0  # DRAM/HBM activity percentage
    fp64_active: float = 0.0  # FP64 pipe activity
    fp32_active: float = 0.0  # FP32 pipe activity
    fp16_active: float = 0.0  # FP16 pipe activity
    pcie_gen: int = 0  # PCIe generation (4, 5, etc.)
    pcie_width: int = 0  # PCIe link width (x16, etc.)


class DCGMPrometheusExporter:
    """
    Exports DCGM GPU metrics to Prometheus format.
    
    Implements the monitoring approach described in Chapter 16 for production
    LLM inference clusters.
    """
    
    def __init__(self, update_interval: float = 5.0):
        """
        Initialize the exporter.
        
        Args:
            update_interval: How often to update metrics (seconds)
        """
        self.update_interval = update_interval
        self.hostname = socket.gethostname()
        self.dcgm_handle = None
        self.dcgm_group = None
        self.dcgm_field_group = None
        self.dcgm_field_ids: Dict[str, int] = {}
        self.dcgm_ready = False
        self.prev_nvlink_errors: Dict[int, int] = {}
        self._prev_nvlink_totals: Dict[int, Tuple[float, float]] = {}
        self._gpu_info_cache: Dict[str, str] = {'hostname': self.hostname}

        if DCGM_AVAILABLE:
            self.dcgm_ready = self._initialize_dcgm()
        
        # Define Prometheus metrics
        self.gpu_utilization = Gauge(
            'dcgm_gpu_utilization_percent',
            'GPU SM utilization percentage',
            ['gpu', 'hostname']
        )
        
        self.gpu_memory_used = Gauge(
            'dcgm_gpu_memory_used_bytes',
            'GPU framebuffer memory used',
            ['gpu', 'hostname']
        )
        
        self.gpu_memory_total = Gauge(
            'dcgm_gpu_memory_total_bytes',
            'GPU total framebuffer memory',
            ['gpu', 'hostname']
        )
        
        self.gpu_memory_utilization = Gauge(
            'dcgm_gpu_memory_utilization_percent',
            'GPU memory utilization percentage',
            ['gpu', 'hostname']
        )
        
        self.gpu_temperature = Gauge(
            'dcgm_gpu_temperature_celsius',
            'GPU temperature in Celsius',
            ['gpu', 'hostname']
        )
        
        self.gpu_power = Gauge(
            'dcgm_gpu_power_watts',
            'GPU power usage in watts',
            ['gpu', 'hostname']
        )
        
        self.gpu_sm_clock = Gauge(
            'dcgm_gpu_sm_clock_mhz',
            'GPU SM clock frequency in MHz',
            ['gpu', 'hostname']
        )
        
        self.gpu_memory_clock = Gauge(
            'dcgm_gpu_memory_clock_mhz',
            'GPU memory clock frequency in MHz',
            ['gpu', 'hostname']
        )
        
        self.pcie_tx_throughput = Gauge(
            'dcgm_pcie_tx_bytes_per_sec',
            'PCIe transmit throughput',
            ['gpu', 'hostname']
        )
        
        self.pcie_rx_throughput = Gauge(
            'dcgm_pcie_rx_bytes_per_sec',
            'PCIe receive throughput',
            ['gpu', 'hostname']
        )
        
        self.nvlink_tx_throughput = Gauge(
            'dcgm_nvlink_tx_bytes_per_sec',
            'NVLink transmit throughput',
            ['gpu', 'hostname']
        )
        
        self.nvlink_rx_throughput = Gauge(
            'dcgm_nvlink_rx_bytes_per_sec',
            'NVLink receive throughput',
            ['gpu', 'hostname']
        )
        
        self.nvlink_errors = Counter(
            'dcgm_nvlink_errors_total',
            'Total NVLink errors',
            ['gpu', 'hostname', 'error_type']
        )
        
        self.encoder_utilization = Gauge(
            'dcgm_gpu_encoder_utilization_percent',
            'GPU video encoder utilization',
            ['gpu', 'hostname']
        )
        
        self.decoder_utilization = Gauge(
            'dcgm_gpu_decoder_utilization_percent',
            'GPU video decoder utilization',
            ['gpu', 'hostname']
        )
        
        # =================================================================
        # Extended Profiling Metrics (DCGM Profiling Fields)
        # =================================================================
        
        # PCIe health metrics
        self.pcie_replay_counter = Counter(
            'dcgm_pcie_replay_total',
            'Total PCIe replay events (link quality indicator)',
            ['gpu', 'hostname']
        )
        
        self.pcie_gen = Gauge(
            'dcgm_pcie_generation',
            'PCIe generation (4, 5, etc.)',
            ['gpu', 'hostname']
        )
        
        self.pcie_width = Gauge(
            'dcgm_pcie_link_width',
            'PCIe link width (16 for x16)',
            ['gpu', 'hostname']
        )
        
        # Error tracking
        self.xid_errors = Counter(
            'dcgm_xid_errors_total',
            'Total GPU XID error events',
            ['gpu', 'hostname', 'xid_code']
        )
        
        self.retired_pages_sbe = Gauge(
            'dcgm_retired_pages_sbe',
            'Retired pages due to single-bit ECC errors',
            ['gpu', 'hostname']
        )
        
        self.retired_pages_dbe = Gauge(
            'dcgm_retired_pages_dbe',
            'Retired pages due to double-bit ECC errors',
            ['gpu', 'hostname']
        )
        
        # SM profiling metrics
        self.sm_active = Gauge(
            'dcgm_sm_active_percent',
            'SM activity percentage',
            ['gpu', 'hostname']
        )
        
        self.sm_occupancy = Gauge(
            'dcgm_sm_occupancy_percent',
            'SM occupancy percentage',
            ['gpu', 'hostname']
        )
        
        # Pipe utilization metrics
        self.tensor_active = Gauge(
            'dcgm_tensor_active_percent',
            'Tensor core activity percentage',
            ['gpu', 'hostname']
        )
        
        self.dram_active = Gauge(
            'dcgm_dram_active_percent',
            'DRAM/HBM activity percentage',
            ['gpu', 'hostname']
        )
        
        self.fp64_active = Gauge(
            'dcgm_fp64_active_percent',
            'FP64 pipe activity percentage',
            ['gpu', 'hostname']
        )
        
        self.fp32_active = Gauge(
            'dcgm_fp32_active_percent',
            'FP32 pipe activity percentage',
            ['gpu', 'hostname']
        )
        
        self.fp16_active = Gauge(
            'dcgm_fp16_active_percent',
            'FP16 pipe activity percentage',
            ['gpu', 'hostname']
        )
        
        # Throughput histogram for latency analysis
        self.kernel_duration = Histogram(
            'dcgm_kernel_duration_seconds',
            'Histogram of kernel execution durations',
            ['gpu', 'hostname'],
            buckets=(0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0)
        )
        
        # System info
        self.gpu_info = Info(
            'dcgm_gpu_info',
            'GPU device information'
        )

    def _initialize_dcgm(self) -> bool:
        """Initialize DCGM handles and watches."""
        try:
            self.dcgm_handle = pydcgm.DcgmHandle()
            self.dcgm_group = pydcgm.DcgmGroup(
                self.dcgm_handle,
                groupId=dcgm_structs.DCGM_GROUP_ALL_GPUS
            )

            # Core metrics
            field_mapping = {
                'gpu_util': dcgm_fields.DCGM_FI_DEV_GPU_UTIL,
                'mem_copy': dcgm_fields.DCGM_FI_DEV_MEM_COPY_UTIL,
                'fb_used': dcgm_fields.DCGM_FI_DEV_FB_USED,
                'fb_total': dcgm_fields.DCGM_FI_DEV_FB_TOTAL,
                'temperature': dcgm_fields.DCGM_FI_DEV_GPU_TEMP,
                'power': dcgm_fields.DCGM_FI_DEV_POWER_USAGE,
                'sm_clock': dcgm_fields.DCGM_FI_DEV_SM_CLOCK,
                'mem_clock': dcgm_fields.DCGM_FI_DEV_MEM_CLOCK,
                'nvlink_total': dcgm_fields.DCGM_FI_DEV_NVLINK_BANDWIDTH_TOTAL,
                'nvlink_errors': dcgm_fields.DCGM_FI_DEV_GPU_NVLINK_ERRORS,
            }
            
            # Extended metrics - PCIe health and errors
            extended_fields = {
                'pcie_replay': getattr(dcgm_fields, 'DCGM_FI_DEV_PCIE_REPLAY_COUNTER', None),
                'xid_errors': getattr(dcgm_fields, 'DCGM_FI_DEV_XID_ERRORS', None),
                'retired_sbe': getattr(dcgm_fields, 'DCGM_FI_DEV_RETIRED_SBE', None),
                'retired_dbe': getattr(dcgm_fields, 'DCGM_FI_DEV_RETIRED_DBE', None),
                'pcie_gen': getattr(dcgm_fields, 'DCGM_FI_DEV_PCIE_LINK_GEN', None),
                'pcie_width': getattr(dcgm_fields, 'DCGM_FI_DEV_PCIE_LINK_WIDTH', None),
            }
            
            # Profiling metrics (DCGM_FI_PROF_* fields) - require profiling mode
            profiling_fields = {
                'sm_active': getattr(dcgm_fields, 'DCGM_FI_PROF_SM_ACTIVE', None),
                'sm_occupancy': getattr(dcgm_fields, 'DCGM_FI_PROF_SM_OCCUPANCY', None),
                'tensor_active': getattr(dcgm_fields, 'DCGM_FI_PROF_PIPE_TENSOR_ACTIVE', None),
                'dram_active': getattr(dcgm_fields, 'DCGM_FI_PROF_DRAM_ACTIVE', None),
                'fp64_active': getattr(dcgm_fields, 'DCGM_FI_PROF_PIPE_FP64_ACTIVE', None),
                'fp32_active': getattr(dcgm_fields, 'DCGM_FI_PROF_PIPE_FP32_ACTIVE', None),
                'fp16_active': getattr(dcgm_fields, 'DCGM_FI_PROF_PIPE_FP16_ACTIVE', None),
            }
            
            # Merge all available fields
            for name, field_id in extended_fields.items():
                if field_id is not None:
                    field_mapping[name] = field_id
            
            for name, field_id in profiling_fields.items():
                if field_id is not None:
                    field_mapping[name] = field_id
            
            self.dcgm_field_ids = field_mapping

            self.dcgm_field_group = pydcgm.DcgmFieldGroup(
                self.dcgm_handle,
                name='prometheus_exporter_fields',
                fieldIds=list(field_mapping.values())
            )

            update_freq_usec = max(int(self.update_interval * 1_000_000), 200_000)
            max_keep_age = max(int(self.update_interval * 4), 30)
            self.dcgm_group.samples.WatchFields(
                self.dcgm_field_group,
                update_freq_usec,
                max_keep_age,
                0
            )

            try:
                dcgm_agent.dcgmUpdateAllFields(self.dcgm_handle.handle, 1)
            except Exception:
                # Best-effort; nothing critical if this fails
                pass

            gpu_ids = self.dcgm_group.GetGpuIds()
            now_ts = time.time()
            for gid in gpu_ids:
                self.prev_nvlink_errors.setdefault(gid, 0)
                self._prev_nvlink_totals.setdefault(gid, (0.0, now_ts))

            return True
        except Exception as exc:  # pragma: no cover - hardware setup specific
            print(f"Warning: Failed to initialize DCGM monitoring: {exc}")
            self.dcgm_handle = None
            self.dcgm_group = None
            self.dcgm_field_group = None
            self.dcgm_field_ids = {}
            return False

    def _extract_field(self, field_map: Dict[int, Any], key: str, default: float = 0.0) -> float:
        """Safely extract the most recent value for the given field key."""
        field_id = self.dcgm_field_ids.get(key)
        if field_id is None:
            return default

        series = field_map.get(field_id)
        if not series or len(series) == 0:
            return default

        latest = series[-1]
        value = getattr(latest, "value", None)
        if value is None or getattr(latest, "isBlank", False):
            return default
        return float(value)

    def collect_dcgm_metrics(self) -> List[GPUMetrics]:
        """
        Gather GPU metrics using DCGM field watches.

        Returns:
            List of GPUMetrics objects, or [] if unavailable.
        """
        if not self.dcgm_ready or self.dcgm_field_group is None:
            return []

        try:
            collection = self.dcgm_group.samples.GetLatest(self.dcgm_field_group)
        except Exception as exc:  # pragma: no cover - hardware setup specific
            print(f"Warning: DCGM metric collection failed: {exc}")
            return []

        metrics: List[GPUMetrics] = []
        timestamp = time.time()

        for gpu_id, field_series in collection.values.items():
            mem_used_mb = self._extract_field(field_series, 'fb_used', 0.0)
            mem_total_mb = self._extract_field(field_series, 'fb_total', 0.0)
            mem_used_bytes = int(mem_used_mb * 1024 * 1024)
            mem_total_bytes = int(mem_total_mb * 1024 * 1024)
            mem_util_pct = (mem_used_bytes / mem_total_bytes * 100.0) if mem_total_bytes else 0.0

            # NVLink throughput: treat counter delta per second as bytes/sec (counter reported in bytes)
            total_counter = self._extract_field(field_series, 'nvlink_total', 0.0)
            prev_total, prev_ts = self._prev_nvlink_totals.get(gpu_id, (total_counter, timestamp))
            elapsed = max(timestamp - prev_ts, 1e-3)
            nvlink_delta = max(0.0, total_counter - prev_total)
            nvlink_bw = nvlink_delta / elapsed if total_counter >= prev_total else 0.0
            self._prev_nvlink_totals[gpu_id] = (total_counter, timestamp)

            # NVLink errors (monotonic counter)
            nvlink_errors = int(self._extract_field(field_series, 'nvlink_errors', 0.0))

            gpu_util = self._extract_field(field_series, 'gpu_util', 0.0)
            mem_copy_util = self._extract_field(field_series, 'mem_copy', 0.0)
            temperature = self._extract_field(field_series, 'temperature', 0.0)
            power_usage = self._extract_field(field_series, 'power', 0.0)
            sm_clock = int(self._extract_field(field_series, 'sm_clock', 0.0))
            memory_clock = int(self._extract_field(field_series, 'mem_clock', 0.0))

            pcie_tx = 0
            pcie_rx = 0
            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
                    pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                    ) * 1024
                    pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                        handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                    ) * 1024
                except Exception:
                    pcie_tx = pcie_rx = 0

            # Extract extended metrics
            pcie_replay = int(self._extract_field(field_series, 'pcie_replay', 0.0))
            xid_errors = int(self._extract_field(field_series, 'xid_errors', 0.0))
            retired_sbe = int(self._extract_field(field_series, 'retired_sbe', 0.0))
            retired_dbe = int(self._extract_field(field_series, 'retired_dbe', 0.0))
            pcie_gen = int(self._extract_field(field_series, 'pcie_gen', 0.0))
            pcie_width = int(self._extract_field(field_series, 'pcie_width', 0.0))
            
            # Extract profiling metrics
            sm_active = self._extract_field(field_series, 'sm_active', 0.0)
            sm_occupancy = self._extract_field(field_series, 'sm_occupancy', 0.0)
            tensor_active = self._extract_field(field_series, 'tensor_active', 0.0)
            dram_active = self._extract_field(field_series, 'dram_active', 0.0)
            fp64_active = self._extract_field(field_series, 'fp64_active', 0.0)
            fp32_active = self._extract_field(field_series, 'fp32_active', 0.0)
            fp16_active = self._extract_field(field_series, 'fp16_active', 0.0)

            metrics.append(GPUMetrics(
                gpu_id=gpu_id,
                gpu_util=gpu_util,
                mem_util=mem_copy_util if mem_copy_util > 0 else mem_util_pct,
                mem_used=mem_used_bytes,
                mem_total=mem_total_bytes,
                temperature=temperature,
                power_usage=power_usage,
                pcie_tx=int(pcie_tx),
                pcie_rx=int(pcie_rx),
                nvlink_tx=int(nvlink_bw),
                nvlink_rx=int(nvlink_bw),
                sm_clock=sm_clock,
                memory_clock=memory_clock,
                encoder_util=0.0,
                decoder_util=0.0,
                nvlink_errors=nvlink_errors,
                # Extended metrics
                pcie_replay_counter=pcie_replay,
                xid_errors=xid_errors,
                retired_pages_sbe=retired_sbe,
                retired_pages_dbe=retired_dbe,
                sm_active=sm_active,
                sm_occupancy=sm_occupancy,
                tensor_active=tensor_active,
                dram_active=dram_active,
                fp64_active=fp64_active,
                fp32_active=fp32_active,
                fp16_active=fp16_active,
                pcie_gen=pcie_gen,
                pcie_width=pcie_width,
            ))

        return metrics
        
    def collect_nvml_metrics(self) -> List[GPUMetrics]:
        """
        Collect GPU metrics using NVML (fallback when DCGM unavailable).
        
        Returns:
            List of GPUMetrics for each GPU
        """
        if not NVML_AVAILABLE:
            return []
            
        metrics = []
        device_count = pynvml.nvmlDeviceGetCount()
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            
            # Get memory info
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            
            # Get temperature
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            
            # Get power
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
            
            # Get clocks
            sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
            mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
            
            # Try to get PCIe throughput (may not be available on all GPUs)
            try:
                pcie_tx = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_TX_BYTES
                ) * 1024  # KB/s to bytes/s
                pcie_rx = pynvml.nvmlDeviceGetPcieThroughput(
                    handle, pynvml.NVML_PCIE_UTIL_RX_BYTES
                ) * 1024
            except pynvml.NVMLError:
                pcie_tx = pcie_rx = 0  # Not supported on this GPU
            
            # Try to get encoder/decoder util (may not be available)
            try:
                encoder_util, _ = pynvml.nvmlDeviceGetEncoderUtilization(handle)
                decoder_util, _ = pynvml.nvmlDeviceGetDecoderUtilization(handle)
            except pynvml.NVMLError:
                encoder_util = decoder_util = 0  # Not supported on this GPU
            
            metrics.append(GPUMetrics(
                gpu_id=i,
                gpu_util=util.gpu,
                mem_util=util.memory,
                mem_used=mem_info.used,
                mem_total=mem_info.total,
                temperature=temp,
                power_usage=power,
                pcie_tx=pcie_tx,
                pcie_rx=pcie_rx,
                nvlink_tx=0,  # NVML doesn't provide NVLink stats easily
                nvlink_rx=0,
                sm_clock=sm_clock,
                memory_clock=mem_clock,
                encoder_util=encoder_util,
                decoder_util=decoder_util,
                nvlink_errors=0
            ))
            
        return metrics
    
    def collect_nvidia_smi_metrics(self) -> List[GPUMetrics]:
        """
        Fallback to nvidia-smi for basic metrics.
        
        Returns:
            List of GPUMetrics for each GPU
        """
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu,utilization.memory,'
                 'memory.used,memory.total,temperature.gpu,power.draw,'
                 'clocks.current.sm,clocks.current.memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                check=True
            )
            
            metrics = []
            for line in result.stdout.strip().split('\n'):
                if not line:
                    continue
                    
                parts = [x.strip() for x in line.split(',')]
                if len(parts) < 9:
                    continue
                    
                try:
                    metrics.append(GPUMetrics(
                        gpu_id=int(parts[0]),
                        gpu_util=float(parts[1]),
                        mem_util=float(parts[2]),
                        mem_used=int(parts[3]) * 1024 * 1024,  # MiB to bytes
                        mem_total=int(parts[4]) * 1024 * 1024,
                        temperature=float(parts[5]),
                        power_usage=float(parts[6]),
                        pcie_tx=0,
                        pcie_rx=0,
                        nvlink_tx=0,
                        nvlink_rx=0,
                        sm_clock=int(parts[7]),
                        memory_clock=int(parts[8]),
                        encoder_util=0,
                        decoder_util=0,
                        nvlink_errors=0
                    ))
                except (ValueError, IndexError) as e:
                    print(f"Error parsing nvidia-smi output: {e}")
                    continue
                    
            return metrics
            
        except subprocess.CalledProcessError as e:
            print(f"Error running nvidia-smi: {e}")
            return []
    
    def update_metrics(self):
        """Update all Prometheus metrics with current GPU stats."""
        # Prefer DCGM if available for richer metrics, else fallback
        metrics = []
        if self.dcgm_ready:
            metrics = self.collect_dcgm_metrics()

        if not metrics:
            if NVML_AVAILABLE:
                metrics = self.collect_nvml_metrics()
            else:
                metrics = self.collect_nvidia_smi_metrics()
        
        if not metrics:
            print("Warning: No GPU metrics available")
            return
        
        # Update Prometheus metrics
        for m in metrics:
            gpu_label = f"gpu{m.gpu_id}"
            
            self.gpu_utilization.labels(gpu=gpu_label, hostname=self.hostname).set(m.gpu_util)
            self.gpu_memory_used.labels(gpu=gpu_label, hostname=self.hostname).set(m.mem_used)
            self.gpu_memory_total.labels(gpu=gpu_label, hostname=self.hostname).set(m.mem_total)
            
            mem_util_pct = (m.mem_used / m.mem_total * 100) if m.mem_total > 0 else 0
            self.gpu_memory_utilization.labels(gpu=gpu_label, hostname=self.hostname).set(mem_util_pct)
            
            self.gpu_temperature.labels(gpu=gpu_label, hostname=self.hostname).set(m.temperature)
            self.gpu_power.labels(gpu=gpu_label, hostname=self.hostname).set(m.power_usage)
            self.gpu_sm_clock.labels(gpu=gpu_label, hostname=self.hostname).set(m.sm_clock)
            self.gpu_memory_clock.labels(gpu=gpu_label, hostname=self.hostname).set(m.memory_clock)
            
            self.pcie_tx_throughput.labels(gpu=gpu_label, hostname=self.hostname).set(m.pcie_tx)
            self.pcie_rx_throughput.labels(gpu=gpu_label, hostname=self.hostname).set(m.pcie_rx)
            
            self.nvlink_tx_throughput.labels(gpu=gpu_label, hostname=self.hostname).set(m.nvlink_tx)
            self.nvlink_rx_throughput.labels(gpu=gpu_label, hostname=self.hostname).set(m.nvlink_rx)
            
            if m.encoder_util > 0:
                self.encoder_utilization.labels(gpu=gpu_label, hostname=self.hostname).set(m.encoder_util)
                self.decoder_utilization.labels(gpu=gpu_label, hostname=self.hostname).set(m.decoder_util)
            
            # Extended metrics - PCIe health
            if m.pcie_gen > 0:
                self.pcie_gen.labels(gpu=gpu_label, hostname=self.hostname).set(m.pcie_gen)
            if m.pcie_width > 0:
                self.pcie_width.labels(gpu=gpu_label, hostname=self.hostname).set(m.pcie_width)
            
            # Error tracking (counters - track deltas)
            if m.retired_pages_sbe > 0:
                self.retired_pages_sbe.labels(gpu=gpu_label, hostname=self.hostname).set(m.retired_pages_sbe)
            if m.retired_pages_dbe > 0:
                self.retired_pages_dbe.labels(gpu=gpu_label, hostname=self.hostname).set(m.retired_pages_dbe)
            
            # Profiling metrics - SM and pipe utilization
            if m.sm_active > 0:
                self.sm_active.labels(gpu=gpu_label, hostname=self.hostname).set(m.sm_active)
            if m.sm_occupancy > 0:
                self.sm_occupancy.labels(gpu=gpu_label, hostname=self.hostname).set(m.sm_occupancy)
            if m.tensor_active > 0:
                self.tensor_active.labels(gpu=gpu_label, hostname=self.hostname).set(m.tensor_active)
            if m.dram_active > 0:
                self.dram_active.labels(gpu=gpu_label, hostname=self.hostname).set(m.dram_active)
            if m.fp64_active > 0:
                self.fp64_active.labels(gpu=gpu_label, hostname=self.hostname).set(m.fp64_active)
            if m.fp32_active > 0:
                self.fp32_active.labels(gpu=gpu_label, hostname=self.hostname).set(m.fp32_active)
            if m.fp16_active > 0:
                self.fp16_active.labels(gpu=gpu_label, hostname=self.hostname).set(m.fp16_active)

            # Increment each GPU's NVLink error counter from its own cumulative
            # source value. Keeping this inside the per-GPU loop prevents the
            # final record from being applied as if it represented every device.
            prev_errors = self.prev_nvlink_errors.get(m.gpu_id, 0)
            if m.nvlink_errors >= prev_errors:
                delta = m.nvlink_errors - prev_errors
            else:
                # Counter reset
                delta = m.nvlink_errors
            if delta > 0:
                self.nvlink_errors.labels(
                    gpu=gpu_label,
                    hostname=self.hostname,
                    error_type='total'
                ).inc(delta)
            self.prev_nvlink_errors[m.gpu_id] = m.nvlink_errors

            if NVML_AVAILABLE:
                try:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(m.gpu_id)
                    name = pynvml.nvmlDeviceGetName(handle)
                    if isinstance(name, bytes):
                        name = name.decode('utf-8', errors='ignore')
                except Exception:
                    name = "unknown"
            else:
                name = "unknown"
            self._gpu_info_cache[f'{gpu_label}_name'] = name

        if self._gpu_info_cache:
            self.gpu_info.info(self._gpu_info_cache)
    
    def run(self):
        """Main loop: continuously collect and export metrics."""
        print(f"Starting DCGM/Prometheus exporter on {self.hostname}")
        print(f"Update interval: {self.update_interval}s")
        
        while True:
            try:
                self.update_metrics()
            except Exception as e:
                print(f"Error updating metrics: {e}")
            
            time.sleep(self.update_interval)


def main():
    parser = argparse.ArgumentParser(
        description='Export DCGM GPU metrics to Prometheus'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=8000,
        help='Port to expose metrics on (default: 8000)'
    )
    parser.add_argument(
        '--interval',
        type=float,
        default=5.0,
        help='Metric update interval in seconds (default: 5.0)'
    )
    
    args = parser.parse_args()
    
    # Start Prometheus HTTP server
    start_http_server(args.port)
    print(f"Prometheus metrics available at http://localhost:{args.port}/metrics")
    
    # Start metric collection
    exporter = DCGMPrometheusExporter(update_interval=args.interval)
    exporter.run()


if __name__ == '__main__':
    main()
