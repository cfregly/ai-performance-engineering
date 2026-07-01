from pathlib import Path

from ch10.flash_attention_common import FLASH_ATTENTION_OUTPUT_TOLERANCE


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_flash_attention_is_tried_before_other_sdpa_backends() -> None:
    source = (REPO_ROOT / "ch10/optimized_flash_attention.py").read_text()
    flash_idx = source.index("SDPBackend.FLASH_ATTENTION")
    efficient_idx = source.index("SDPBackend.EFFICIENT_ATTENTION")
    assert flash_idx < efficient_idx
    assert "[SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]" not in source
    assert "major >= 10" in source
    assert "FAIL FAST: FlashAttention required for ch10" in source


def test_external_flashattention_engines_are_probed_in_preference_order() -> None:
    source = (REPO_ROOT / "ch10/optimized_flash_attention.py").read_text()

    flash3_idx = source.index("flash_attn_3.flash_attn_interface")
    flash2_idx = source.index("flash_attn.flash_attn_interface")

    assert flash3_idx < flash2_idx


def test_flash_attention_pair_shares_verification_tolerance() -> None:
    baseline = (REPO_ROOT / "ch10/baseline_flash_attention.py").read_text()
    optimized = (REPO_ROOT / "ch10/optimized_flash_attention.py").read_text()

    assert FLASH_ATTENTION_OUTPUT_TOLERANCE == (5e-2, 5e-2)
    assert "output_tolerance=FLASH_ATTENTION_OUTPUT_TOLERANCE" in baseline
    assert "output_tolerance=FLASH_ATTENTION_OUTPUT_TOLERANCE" in optimized
    assert "output_tolerance=(0.2, 2.0)" not in baseline


def test_flash_attention_pair_reuses_verification_buffers() -> None:
    for name in ("baseline_flash_attention.py", "optimized_flash_attention.py"):
        source = (REPO_ROOT / f"ch10/{name}").read_text()
        setup_section = source.split("def setup", maxsplit=1)[1].split(
            "def _manual_attention" if name.startswith("baseline") else "def benchmark_fn",
            maxsplit=1,
        )[0]
        capture_section = source.split("def capture_verification_payload", maxsplit=1)[1].split(
            "def teardown",
            maxsplit=1,
        )[0]
        teardown_section = source.split("def teardown", maxsplit=1)[1].split(
            "def get_config",
            maxsplit=1,
        )[0]

        assert "self._verify_output_buffer: Optional[torch.Tensor] = None" in source
        assert "self._verify_output_buffer = torch.empty_like(self.input, dtype=torch.float32)" in setup_section
        assert "self._verify_output_buffer.copy_(self.output)" in capture_section
        assert "output=self._verify_output_buffer" in capture_section
        assert "self.output.detach().float().clone()" not in capture_section
        assert "self._verify_output_buffer = None" in teardown_section


def test_baseline_flash_attention_builds_causal_mask_directly() -> None:
    source = (REPO_ROOT / "ch10/baseline_flash_attention.py").read_text()
    setup_source = source.split("def setup", maxsplit=1)[1].split(
        "def _manual_attention",
        maxsplit=1,
    )[0]
    manual_source = source.split("def _manual_attention", maxsplit=1)[1].split(
        "def benchmark_fn",
        maxsplit=1,
    )[0]

    assert "torch.ones(" not in setup_source
    assert ".triu(" not in setup_source
    assert "pos = torch.arange(self.seq_len, device=self.device)" in setup_source
    assert "self._causal_mask = pos.unsqueeze(0) > pos.unsqueeze(1)" in setup_source
    assert "attn_weights.masked_fill_(self._causal_mask, float(\"-inf\"))" in manual_source
    assert (
        "attn_weights = attn_weights.masked_fill(self._causal_mask, float(\"-inf\"))"
        not in manual_source
    )


def test_optimized_flash_attention_reuses_projection_buffers_in_inference() -> None:
    source = (REPO_ROOT / "ch10/optimized_flash_attention.py").read_text()
    tiled_module = source.split("class TiledAttentionModule", maxsplit=1)[1].split(
        "class OptimizedFlashAttentionBenchmark",
        maxsplit=1,
    )[0]

    assert "self._q_buffer: Optional[torch.Tensor] = None" in tiled_module
    assert "self._k_buffer: Optional[torch.Tensor] = None" in tiled_module
    assert "self._v_buffer: Optional[torch.Tensor] = None" in tiled_module
    assert "self._output_buffer: Optional[torch.Tensor] = None" in tiled_module
    assert "self._q_forward_view: Optional[torch.Tensor] = None" in tiled_module
    assert "self._k_forward_view: Optional[torch.Tensor] = None" in tiled_module
    assert "self._v_forward_view: Optional[torch.Tensor] = None" in tiled_module
    assert "self._output_forward_view: Optional[torch.Tensor] = None" in tiled_module
    assert "self._q_proj_weight_t: Optional[torch.Tensor] = None" in tiled_module
    assert "self._k_proj_weight_t: Optional[torch.Tensor] = None" in tiled_module
    assert "self._v_proj_weight_t: Optional[torch.Tensor] = None" in tiled_module
    assert "self._out_proj_weight_t: Optional[torch.Tensor] = None" in tiled_module
    assert "def cache_weight_views(self) -> None:" in tiled_module
    assert "self._q_proj_weight_t = self.q_proj.weight.t()" in tiled_module
    assert "self._out_proj_weight_t = self.out_proj.weight.t()" in tiled_module
    assert "def _ensure_projection_buffers(" in tiled_module
    assert "def prepare_projection_buffers(self, x: torch.Tensor) -> None:" in tiled_module
    assert "self._q_forward_view," in tiled_module
    assert "self._output_forward_view," in tiled_module
    assert "def forward_prepared(self, x: torch.Tensor, is_causal: bool = False)" in tiled_module
    assert "q, k, v = self._project_qkv_prepared(x)" in tiled_module
    assert "return self._project_output_prepared(attn_output.transpose(1, 2).contiguous())" in tiled_module
    assert "if torch.is_grad_enabled():" in tiled_module
    assert "q = torch.matmul(x, self._q_proj_weight_t, out=q_buffer)" in tiled_module
    assert "k = torch.matmul(x, self._k_proj_weight_t, out=k_buffer)" in tiled_module
    assert "v = torch.matmul(x, self._v_proj_weight_t, out=v_buffer)" in tiled_module
    assert "return torch.matmul(merged, self._out_proj_weight_t, out=output_buffer)" in tiled_module
    project_qkv = tiled_module.split("def _project_qkv", maxsplit=1)[1].split(
        "def _project_output",
        maxsplit=1,
    )[0]
    project_output = tiled_module.split("def _project_output", maxsplit=1)[1].split(
        "def forward",
        maxsplit=1,
    )[0]
    assert "self.q_proj.weight.t()" not in project_qkv
    assert "self.k_proj.weight.t()" not in project_qkv
    assert "self.v_proj.weight.t()" not in project_qkv
    assert "self.out_proj.weight.t()" not in project_output
