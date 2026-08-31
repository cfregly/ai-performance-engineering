# Contributing to AI Systems Performance Engineering

Thank you for your interest in contributing to the AI Systems Performance Engineering repository! This guide will help you get started with contributing code, documentation, examples, and improvements.

## How to Contribute

We welcome contributions from the community in many forms:

- **Code Examples**: New CUDA kernels, PyTorch optimizations, performance scripts
- **Documentation**: Improvements to README files, code comments, tutorials
- **Performance Optimizations**: Better algorithms, memory optimizations, profiling tools
- **Bug Fixes**: Issues with existing code, compatibility problems
- **Architecture Support**: Extend Blackwell workflows or add tooling for new GPU families
- **Testing**: Unit tests, performance benchmarks, validation scripts

## Getting Started

### Prerequisites

- NVIDIA GPU with CUDA support
- Python 3.12 for the pinned repository environment
- The supported CUDA/PyTorch stack documented in [Environment and Configuration](docs/environment.md)
- Git

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/your-username/ai-performance-engineering.git
cd ai-performance-engineering

# Create a new branch for your contribution
git checkout -b feature/your-feature-name

# Create an isolated environment from the code directory
cd code
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements_latest.txt
```

## Contribution Guidelines

### Code Style

- **Python**: Follow PEP 8 style guidelines
- **CUDA**: Use consistent naming conventions and proper error handling
- **Shell Scripts**: Use bash with proper error handling (`set -e`)
- **Comments**: Add clear, descriptive comments for complex logic

### File Organization

- **New Examples**: Place in appropriate chapter directory (`code/ch01/` through `code/ch20/`, or `code/labs/`)
- **Tools**: Add to the appropriate `code/core/` package
- **Scripts**: Add to `code/core/scripts/` or the relevant chapter/lab directory
- **Documentation**: Update relevant README files

### Architecture Support

B200/GB200 are **SM100**; B300/GB300 are **SM103**. SM120/121 have separate feature constraints. Use the shared `code/core/common/cuda_arch.mk` selection and the CUDA 13 toolchain described in [the environment guide](docs/environment.md). Preserve explicit unsupported-target checks; do not rewrite one architecture as another to make a build proceed. The B200 dependency baseline does not qualify B300/GB300. See [NVIDIA compute capabilities](https://developer.nvidia.com/cuda/gpus).

## Development Workflow

### 1. Choose Your Contribution Type

#### **Code Examples**
- Create new CUDA kernels or PyTorch optimizations
- Add performance profiling scripts
- Implement new algorithms or techniques

#### **Documentation**
- Improve README files with better explanations
- Add code comments and docstrings
- Create tutorials or guides

#### **Performance Optimizations**
- Optimize existing code for better performance
- Add new profiling tools
- Improve memory usage or compute efficiency

### 2. Development Process

```bash
# Make your changes
# Test your code thoroughly

# From code/, run the repository suite (GPU requirements are explicit skips)
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q -ra -o timeout=120

# Check only files touched by your change with the repository configuration
python -m ruff check core/benchmark/comparison.py
```

### 3. Testing Your Changes

#### **Performance Testing**
```bash
# From code/, discover the exact benchmark pair
python -m cli.aisp bench list-targets --chapter ch01

# On the supported target, check correctness before accepting measurements
python -m cli.aisp bench verify --help
python -m cli.aisp bench run --targets ch01:performance --profile minimal
```

Use the [profiling guide](docs/tooling-and-profiling.md) for current capture
commands and output analysis. A help invocation or source test does not execute
a GPU benchmark, and a historical expectation file does not qualify new source.

#### **Compatibility Testing**
- Record the exact GPU SKU/capability, driver, toolkit, PyTorch and Triton versions.
- Validate B200 and B300 separately; do not pool their timing or correctness evidence.
- Use the CUDA 13 B200 baseline or an independently validated target-specific stack.
- Run full-output numerical checks and applicable sanitizers on actual hardware.

### 4. Submitting Your Contribution

```bash
# Add your changes
git add .

# Commit with descriptive message
git commit -m "Add new CUDA kernel for memory optimization

- Implements coalesced memory access pattern
- Targets NVIDIA Blackwell B200/B300
- Includes performance benchmarks
- Adds comprehensive documentation"

# Push to your fork
git push origin feature/your-feature-name
```

## Pull Request Guidelines

### Before Submitting

- [ ] **Test thoroughly** on the applicable hardware; label CPU/source checks, simulations and hardware checks separately
- [ ] **Update documentation** if needed
- [ ] **Add comments** for complex code
- [ ] **Include performance benchmarks** for optimizations
- [ ] **Follow naming conventions** and code style
- [ ] **Update relevant README files**

### Pull Request Template

```markdown
## Description
Brief description of your changes

## Type of Change
- [ ] New feature (code example, optimization)
- [ ] Bug fix
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Blackwell workflow improvement

## Testing
- [ ] Tested on the exact target: B200/GB200 (SM100) or B300/GB300 (SM103); attach separate receipts
- [ ] Performance benchmarks included
- [ ] Documentation updated

## Performance Impact
- **Before**: [baseline metrics]
- **After**: [improved metrics]
- **Improvement**: [percentage/description]

## Additional Notes
Any additional context or considerations
```

## Architecture Guidelines

### Adding New GPU Support

When adding support for new GPU architectures:

1. **Update architecture detection scripts**
2. **Add new architecture constants**
3. **Test on target hardware**
4. **Update documentation**

### Extending Beyond Blackwell

If you experiment with additional architectures, document the changes clearly and avoid regressing the default Blackwell workflow. Consider maintaining separate branches for architecture-specific divergences to keep `main` lean.

## Performance Contribution Guidelines

### Benchmarking Standards

- **Baseline**: Always include baseline performance
- **Multiple Runs**: Run benchmarks multiple times
- **Hardware Specs**: Document test hardware
- **Environment**: Specify CUDA/PyTorch versions

### Example Benchmark Format

Use the existing `BaseBenchmark` interfaces and a complete baseline/optimized
pair, for example [the Chapter 1 baseline](code/ch01/baseline_performance.py).
Provide deterministic equivalent inputs, an independent full-output reference,
and an explicit numerical error policy. Let the harness own warmup, timing and
post-timing verification. CUDA events must measure the executing stream or join
all participating streams before stopping. Never time an empty loop or count
requested operations that did not execute.

From `code/`, discover and run the pair on its supported target:

```bash
python -m cli.aisp bench list-targets --chapter ch01
python -m cli.aisp bench run --targets ch01:performance --profile minimal
```

Preserve individual measurements and all validation failures; do not manufacture
samples from a summary statistic or use profiler-overhead timings as ordinary
latency. An unchanged source-level contract does not certify a new GPU run.

## Bug Reports

### Reporting Issues

When reporting bugs, please include:

- **Hardware**: GPU model, driver version
- **Software**: CUDA version, PyTorch version
- **Steps**: Clear reproduction steps
- **Expected vs Actual**: What you expected vs what happened
- **Logs**: Error messages and logs

### Issue Template

```markdown
## Bug Description
Clear description of the issue

## Steps to Reproduce
1. Step 1
2. Step 2
3. Step 3

## Expected Behavior
What you expected to happen

## Actual Behavior
What actually happened

## Environment
- GPU: [Model]
- CUDA: [Version]
- PyTorch: [Version]
- OS: [Version]

## Additional Context
Any other relevant information
```

## Documentation Contributions

### README Updates

When updating documentation:

- **Clarity**: Make explanations clear and concise
- **Examples**: Include practical code examples
- **Links**: Add relevant links and references
- **Structure**: Maintain consistent formatting

### Code Comments

- **Purpose**: Explain what the code does
- **Parameters**: Document function parameters
- **Returns**: Document return values
- **Complexity**: Explain complex algorithms

## Contribution Ideas

### High-Priority Areas

- **New CUDA Kernels**: Optimized implementations
- **PyTorch Optimizations**: Framework-specific improvements
- **Profiling Tools**: Better performance analysis
- **Architecture Support**: New GPU compatibility
- **Documentation**: Tutorials and guides

### Example Contributions

- **Memory Optimization**: New memory access patterns
- **Kernel Fusion**: Combining multiple operations
- **Tensor Core Usage**: Optimized matrix operations
- **Stream Management**: Better asynchronous execution
- **Distributed Training**: Multi-GPU optimizations

## Getting Help

### Community Resources

- **Issues**: Use GitHub issues for questions
- **Discussions**: Start discussions for ideas
- **Meetups**: Join our monthly meetups
- **YouTube**: Check our video tutorials

### Contact

- **GitHub Issues**: For bugs and feature requests
- **Discussions**: For questions and ideas
- **Email**: For private or sensitive matters

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project ([Apache License 2.0](LICENSE)).

## Recognition

Contributors will be recognized in:

- **README.md**: For significant contributions
- **Release Notes**: For each release
- **Documentation**: In relevant sections
- **Community**: In meetups and presentations

---

Thank you for contributing to the AI Performance Engineering community.
