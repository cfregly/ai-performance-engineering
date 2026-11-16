#!/usr/bin/env python3
"""
Integration tests to verify consistency across all interfaces.

Tests that CLI, MCP, and Python API expose their supported functionality
and return consistent results.

Run with: pytest tests/test_interface_consistency.py -v
"""

import pytest


# =============================================================================
# Test: PerformanceEngine domains are complete
# =============================================================================

class TestEngineDomains:
    """Verify the 10-domain model is fully implemented."""
    
    def test_all_domains_exist(self):
        """All 10 domains should be accessible."""
        from core.engine import get_engine, DOMAINS
        
        engine = get_engine()
        
        expected_domains = {
            "gpu", "system", "profile", "analyze", "optimize",
            "distributed", "inference", "benchmark", "ai", "export"
        }
        
        assert set(DOMAINS) == expected_domains
        
        for domain in expected_domains:
            assert hasattr(engine, domain), f"Missing domain: {domain}"
            domain_obj = getattr(engine, domain)
            assert domain_obj is not None, f"Domain {domain} is None"
    
    def test_gpu_domain_methods(self):
        """GPU domain should have core methods."""
        from core.engine import get_engine
        
        gpu = get_engine().gpu
        
        required_methods = ["info", "topology", "power", "bandwidth"]
        for method in required_methods:
            assert hasattr(gpu, method), f"GPU domain missing method: {method}"
    
    def test_system_domain_methods(self):
        """System domain should have core methods."""
        from core.engine import get_engine
        
        system = get_engine().system
        
        required_methods = [
            "software",
            "dependencies",
            "context",
            "capabilities",
            "parameters",
            "container",
            "cpu_memory",
            "env",
            "network",
        ]
        for method in required_methods:
            assert hasattr(system, method), f"System domain missing method: {method}"
    
    def test_profile_domain_methods(self):
        """Profile domain should have core methods."""
        from core.engine import get_engine
        
        profile = get_engine().profile
        
        required_methods = ["flame_graph", "memory_timeline", "kernels", "roofline"]
        for method in required_methods:
            assert hasattr(profile, method), f"Profile domain missing method: {method}"
    
    def test_analyze_domain_methods(self):
        """Analyze domain should have core methods."""
        from core.engine import get_engine
        
        analyze = get_engine().analyze
        
        required_methods = ["bottlenecks", "pareto", "scaling", "whatif", "stacking"]
        for method in required_methods:
            assert hasattr(analyze, method), f"Analyze domain missing method: {method}"
    
    def test_optimize_domain_methods(self):
        """Optimize domain should have core methods."""
        from core.engine import get_engine
        
        optimize = get_engine().optimize
        
        required_methods = ["recommend", "roi", "techniques"]
        for method in required_methods:
            assert hasattr(optimize, method), f"Optimize domain missing method: {method}"
    
    def test_distributed_domain_methods(self):
        """Distributed domain should have core methods."""
        from core.engine import get_engine
        
        distributed = get_engine().distributed
        
        required_methods = ["plan", "nccl"]
        for method in required_methods:
            assert hasattr(distributed, method), f"Distributed domain missing method: {method}"
    
    def test_inference_domain_methods(self):
        """Inference domain should have core methods."""
        from core.engine import get_engine
        
        inference = get_engine().inference
        
        required_methods = ["vllm_config", "quantization", "deploy", "estimate"]
        for method in required_methods:
            assert hasattr(inference, method), f"Inference domain missing method: {method}"


# =============================================================================
# Test: MCP tools map to Engine
# =============================================================================

class TestMCPToolsConsistency:
    """Verify MCP tools align with Engine domains."""
    
    def test_mcp_tool_count(self):
        """MCP should stay within the current consolidated tool budget."""
        from mcp.mcp_server import TOOLS
        
        tool_count = len(TOOLS)
        assert 90 <= tool_count <= 120, f"Expected 90-120 tools, got {tool_count}"
    
    def test_mcp_tools_have_no_prefix(self):
        """All MCP tools should be unprefixed."""
        from mcp.mcp_server import TOOLS
        
        for name in TOOLS:
            assert not name.startswith("aisp_"), f"Tool {name} still uses aisp_ prefix"
    
    def test_core_tools_exist(self):
        """Core tools should exist."""
        from mcp.mcp_server import TOOLS
        
        required_tools = [
            "status",
            "triage", 
            "suggest_tools",
            "job_status",
            "gpu_info",
            "system_software",
            "analyze_bottlenecks",
            "optimize",
            "recommend",
            "ask",
        ]
        
        for tool in required_tools:
            assert tool in TOOLS, f"Missing core tool: {tool}"
    
    def test_consolidated_tools_not_present(self):
        """Truly consolidated tools should not exist (merged into others)."""
        from mcp.mcp_server import TOOLS
        
        # These were truly merged into other tools
        merged_tools = [
            "help",           # Merged into suggest_tools
            "hf_search",      # Merged into hf
            "hf_trending",    # Merged into hf
            "hf_download",    # Merged into hf
            "available_benchmarks",  # Merged into benchmark_targets
        ]
        
        for tool in merged_tools:
            assert tool not in TOOLS, f"Merged tool still exists: {tool}"

    def test_ncu_workload_types_match_nsight_metrics(self):
        """NCU workload_type enums should be supported by NsightAutomation."""
        from mcp.mcp_server import TOOLS
        from core.profiling.nsight_automation import NsightAutomation

        schema = TOOLS["profile_ncu"].input_schema
        workload_schema = schema.get("properties", {}).get("workload_type", {})
        enum_values = set(workload_schema.get("enum", []) or [])
        allowed = set(NsightAutomation.METRIC_SETS.keys())

        assert enum_values, "profile_ncu.workload_type enum is empty"
        assert enum_values.issubset(allowed), (
            "profile_ncu workload_type enum has unsupported values"
        )
        default_value = workload_schema.get("default")
        if default_value is not None:
            assert default_value in enum_values, "workload_type default not in enum"

    def test_mcp_handlers_registered(self):
        """All registered tools should have handlers."""
        from mcp.mcp_server import TOOLS, HANDLERS
        
        for name in TOOLS:
            assert name in HANDLERS, f"Tool {name} missing handler"


# =============================================================================
# Test: CLI commands map to Engine
# =============================================================================

class TestCLIConsistency:
    """Verify CLI commands align with Engine domains."""
    
    def test_cli_imports(self):
        """CLI should import without errors."""
        try:
            from cli import aisp
            assert aisp.app is not None
        except Exception as e:
            pytest.fail(f"CLI import failed: {e}")
    
    def test_cli_has_main_app(self):
        """CLI should have main typer app."""
        from cli import aisp
        
        assert hasattr(aisp, 'app')


# =============================================================================
# Test: Cross-interface result consistency
# =============================================================================

class TestResultConsistency:
    """Verify results are consistent across interfaces."""
    
    @pytest.fixture
    def mock_gpu_info(self):
        """Mock GPU info response."""
        return {
            "gpus": [
                {
                    "name": "NVIDIA H100",
                    "memory_total_gb": 80.0,
                    "memory_used_gb": 10.0,
                    "temperature_c": 45,
                    "power_w": 300,
                    "utilization_pct": 50,
                }
            ],
            "count": 1,
        }
    
    def test_engine_returns_dict(self):
        """Engine methods should return dictionaries."""
        from core.engine import get_engine
        
        engine = get_engine()
        
        # Status should return a dict
        result = engine.status()
        assert isinstance(result, dict)
        
        # GPU info should return a dict
        result = engine.gpu.info()
        assert isinstance(result, dict)
    
    def test_mcp_tools_return_dict(self):
        """MCP tools should return dictionaries."""
        from mcp.mcp_server import HANDLERS
        
        # Test a few handlers
        for tool_name in ["status", "triage"]:
            if tool_name in HANDLERS:
                result = HANDLERS[tool_name]({})
                assert isinstance(result, dict), f"{tool_name} should return dict"


# =============================================================================
# Test: Domain naming consistency
# =============================================================================

class TestNamingConsistency:
    """Verify naming is consistent across interfaces."""
    
    def test_domain_names_match(self):
        """Domain names should be consistent."""
        from core.engine import DOMAINS
        
        expected = ["gpu", "system", "profile", "analyze", "optimize",
                   "distributed", "inference", "benchmark", "ai", "export"]
        
        assert list(DOMAINS) == expected
    
    def test_mcp_naming_convention(self):
        """MCP tools should follow {domain}_{operation} pattern."""
        from mcp.mcp_server import TOOLS
        from core.engine import DOMAINS
        
        # Check that domain tools follow the pattern
        domain_tools = [t for t in TOOLS if any(f"{d}_" in t for d in DOMAINS)]
        
        # Should have multiple domain-prefixed tools
        assert len(domain_tools) >= 20, "Not enough domain-prefixed tools"


# =============================================================================
# Test: API documentation consistency
# =============================================================================

class TestDocumentationConsistency:
    """Verify documentation is consistent."""
    
    def test_engine_has_docstrings(self):
        """Engine domains should have docstrings."""
        from core.engine import get_engine
        
        engine = get_engine()
        
        for domain_name in ["gpu", "system", "profile", "analyze", "optimize"]:
            domain = getattr(engine, domain_name)
            assert domain.__doc__ is not None, f"Domain {domain_name} missing docstring"
    
    def test_mcp_tools_have_descriptions(self):
        """MCP tools should have descriptions."""
        from mcp.mcp_server import TOOLS
        
        for name, tool in TOOLS.items():
            assert tool.description, f"Tool {name} missing description"
            assert len(tool.description) > 20, f"Tool {name} description too short"


# =============================================================================
# Test: Async job handling
# =============================================================================

class TestAsyncJobHandling:
    """Verify async job handling works across interfaces."""
    
    def test_job_status_tool_exists(self):
        """job_status should exist."""
        from mcp.mcp_server import TOOLS
        
        assert "job_status" in TOOLS
    
    def test_job_status_requires_job_id(self):
        """job_status should require job_id parameter."""
        from mcp.mcp_server import TOOLS
        
        schema = TOOLS["job_status"].input_schema
        required = schema.get("required", [])
        
        assert "job_id" in required, "job_id should be required"


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
