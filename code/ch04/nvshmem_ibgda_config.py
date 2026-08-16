#!/usr/bin/env python3
"""NVSHMEM IBGDA (GPUDirect Async) configuration helper for Blackwell.

Provides optimal NVSHMEM environment variable configuration for InfiniBand
GPUDirect Async, enabling sub-microsecond GPU-to-GPU communication.

Reference: NVIDIA NVSHMEM 2.7+ documentation
"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

from core.utils.logger import get_logger

logger = get_logger(__name__)


class NVSHMEMIBGDAConfig:
    """NVSHMEM IBGDA configuration manager."""
    
    # Optimal IBGDA settings for Blackwell
    IBGDA_BASE_CONFIG = {
        # Enable IBGDA
        'NVSHMEM_IB_ENABLE_IBGDA': '1',
        
        # GPU-based NIC handler (remove CPU from critical path)
        'NVSHMEM_IBGDA_NIC_HANDLER': 'gpu',
        
        # Force GPU memory for NIC buffers (bypass CPU memory)
        'NVSHMEM_IBGDA_FORCE_NIC_BUF_MEMTYPE': 'gpumem',
        
        # Enable multi-port for dual-rail InfiniBand
        'NVSHMEM_IBGDA_ENABLE_MULTI_PORT': '1',
        
        # Number of requests in batch (tune for latency vs bandwidth)
        'NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH': '1',  # Low latency
        
        # InfiniBand settings
        'NVSHMEM_IB_GID_INDEX': '3',  # RoCE v2 GID index
        'NVSHMEM_IB_TRAFFIC_CLASS': '0',
        
        # Symmetric heap size (adjust based on workload)
        'NVSHMEM_SYMMETRIC_SIZE': '1G',  # 1GB per GPU
        
        # Debug/info (disable for production)
        'NVSHMEM_DEBUG': 'WARN',  # Change to INFO for debugging
        
        # Bootstrap method
        'NVSHMEM_BOOTSTRAP': 'MPI',  # or 'PMI'
    }
    
    # Performance tuning for different message sizes
    TUNING_PROFILES = {
        'low_latency': {
            'NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH': '1',
            'NVSHMEM_IB_NUM_QPS': '1',
            'description': 'Optimized for <1KB messages, sub-microsecond latency'
        },
        'balanced': {
            'NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH': '4',
            'NVSHMEM_IB_NUM_QPS': '2',
            'description': 'Balanced for 1KB-64KB messages'
        },
        'high_bandwidth': {
            'NVSHMEM_IBGDA_NUM_REQUESTS_IN_BATCH': '16',
            'NVSHMEM_IB_NUM_QPS': '4',
            'description': 'Optimized for >64KB messages, maximum bandwidth'
        },
    }
    
    def __init__(self, profile: str = 'balanced'):
        """Initialize IBGDA configuration.
        
        Args:
            profile: Performance profile ('low_latency', 'balanced', 'high_bandwidth')
        """
        self.profile = profile
        self.config = self.IBGDA_BASE_CONFIG.copy()
        
        # Apply profile-specific settings
        if profile in self.TUNING_PROFILES:
            self.config.update(self.TUNING_PROFILES[profile])
            logger.info(f"Using profile: {profile}")
            logger.info(f"  {self.TUNING_PROFILES[profile]['description']}")
    
    def validate_environment(self) -> Dict[str, bool]:
        """Validate IBGDA requirements.
        
        Returns:
            checks: Dict of validation results
        """
        checks = {}
        
        # Check InfiniBand
        try:
            result = subprocess.run(['ibstat'], capture_output=True, text=True)
            checks['infiniband'] = result.returncode == 0
            if checks['infiniband']:
                logger.info("✓ InfiniBand detected")
            else:
                logger.warning("✗ InfiniBand not detected (ibstat failed)")
        except FileNotFoundError:
            checks['infiniband'] = False
            logger.warning("✗ InfiniBand tools not installed")
        
        # Check NVSHMEM version
        try:
            nvshmem_home = os.environ.get('NVSHMEM_HOME', '')
            if nvshmem_home:
                # Try to find version file
                version_file = Path(nvshmem_home) / 'include' / 'nvshmem_version.h'
                if version_file.exists():
                    checks['nvshmem'] = True
                    logger.info(f"✓ NVSHMEM found at {nvshmem_home}")
                else:
                    checks['nvshmem'] = False
                    logger.warning(f"✗ NVSHMEM_HOME set but version.h not found")
            else:
                checks['nvshmem'] = False
                logger.warning("✗ NVSHMEM_HOME not set")
        except Exception as e:
            checks['nvshmem'] = False
            logger.warning(f"✗ NVSHMEM check failed: {e}")
        
        # Check GPUDirect RDMA
        try:
            nvidia_peermem = Path('/sys/module/nvidia_peermem')
            checks['gpu_direct_rdma'] = nvidia_peermem.exists()
            if checks['gpu_direct_rdma']:
                logger.info("✓ GPUDirect RDMA enabled (nvidia_peermem loaded)")
            else:
                logger.warning("✗ GPUDirect RDMA not enabled (nvidia_peermem not loaded)")
        except Exception:
            checks['gpu_direct_rdma'] = False
        
        # Check CUDA
        try:
            import torch
            checks['cuda'] = torch.cuda.is_available()
            if checks['cuda']:
                logger.info(f"✓ CUDA available: {torch.version.cuda}")
            else:
                logger.warning("✗ CUDA not available")
        except ImportError:
            checks['cuda'] = False
            logger.warning("✗ PyTorch not installed")
        
        return checks
    
    def export_config(self, apply: bool = False) -> str:
        """Export configuration as shell commands.
        
        Args:
            apply: If True, apply to current environment
        
        Returns:
            config_str: Shell export commands
        """
        lines = [
            "# NVSHMEM IBGDA Configuration",
            f"# Profile: {self.profile}",
            ""
        ]
        
        for key, value in self.config.items():
            if key == 'description':
                continue
            lines.append(f"export {key}={value}")
            
            if apply:
                os.environ[key] = str(value)
        
        config_str = '\n'.join(lines)
        
        if apply:
            logger.info("Configuration applied to environment")
        
        return config_str
    
    def print_config(self):
        """Print configuration to console."""
        print(f"\n{'='*60}")
        print(f"NVSHMEM IBGDA Configuration - {self.profile}")
        print(f"{'='*60}\n")
        print(self.export_config(apply=False))
        print(f"\n{'='*60}\n")
    
    def save_config(self, output_path: Path):
        """Save configuration to shell script.
        
        Args:
            output_path: Path to save configuration script
        """
        config_str = self.export_config(apply=False)
        
        with open(output_path, 'w') as f:
            f.write("#!/bin/bash\n")
            f.write(f"# NVSHMEM IBGDA Configuration\n")
            f.write(f"# Generated by nvshmem_ibgda_config.py\n\n")
            f.write(config_str)
            f.write("\n")
        
        # Make executable
        output_path.chmod(0o755)
        
        logger.info(f"Configuration saved to {output_path}")
        print(f"\nTo apply: source {output_path}")
    
    def validate_and_report(self) -> bool:
        """Validate environment and print report.
        
        Returns:
            all_pass: True if all checks pass
        """
        print(f"\n{'='*60}")
        print("NVSHMEM IBGDA Environment Validation")
        print(f"{'='*60}\n")
        
        checks = self.validate_environment()
        
        all_pass = all(checks.values())
        
        if all_pass:
            print("\n✓ All checks passed! IBGDA ready to use.\n")
        else:
            print("\n✗ Some checks failed. See warnings above.\n")
            print("IBGDA requirements:")
            print("  1. InfiniBand HCA with GPUDirect RDMA support")
            print("  2. NVSHMEM 2.7+ with IBGDA support")
            print("  3. nvidia_peermem kernel module loaded")
            print("  4. Proper InfiniBand configuration (ibstat)")
            print()
        
        return all_pass


def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="NVSHMEM IBGDA Configuration Helper",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Print configuration
  python nvshmem_ibgda_config.py --profile low_latency
  
  # Save to file
  python nvshmem_ibgda_config.py --profile balanced --output ibgda_config.sh
  
  # Validate environment
  python nvshmem_ibgda_config.py --validate
  
  # Apply to current environment
  python nvshmem_ibgda_config.py --profile low_latency --apply
        """
    )
    
    parser.add_argument(
        '--profile', type=str, default='balanced',
        choices=['low_latency', 'balanced', 'high_bandwidth'],
        help='Performance tuning profile'
    )
    parser.add_argument(
        '--output', type=Path,
        help='Save configuration to shell script'
    )
    parser.add_argument(
        '--validate', action='store_true',
        help='Validate environment and print report'
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Apply configuration to current environment'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config = NVSHMEMIBGDAConfig(profile=args.profile)
    
    # Validate if requested
    if args.validate:
        config.validate_and_report()
        return
    
    # Print configuration
    config.print_config()
    
    # Save if requested
    if args.output:
        config.save_config(args.output)
    
    # Apply if requested
    if args.apply:
        config.export_config(apply=True)
        print("\n✓ Configuration applied to current environment")
        print("  Run NVSHMEM applications in this shell\n")
    
    # Print expected performance
    print("Expected Performance with IBGDA:")
    print("  - Sub-1KB puts: ~9.5× faster than CPU-proxy")
    print("  - Register-level nvshmem_p: ~180 MOPS")
    print("  - Latency: Sub-microsecond for small messages")
    print()


if __name__ == '__main__':
    main()
