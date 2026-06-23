import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_nemotron_webgpu_limit_helper_selects_encoder_variant_by_storage_buffer_limit():
    script = """
        import {
          checkNemotronWebGpuLimits,
          nemotronModelAssetUrl,
          selectNemotronEncoderVariantForLimit,
        } from './app/static/nemotron/nemotron-engine.mjs';

        const standard = selectNemotronEncoderVariantForLimit(25);
        if (standard.file !== 'encoder_fp16.onnx' || standard.minStorageBuffers !== 25) {
          throw new Error(`unexpected standard variant: ${JSON.stringify(standard)}`);
        }

        const limit16 = checkNemotronWebGpuLimits({ maxStorageBuffersPerShaderStage: 16 });
        if (limit16.ok !== true || limit16.variant.file !== 'encoder_fp16_concat16.onnx') {
          throw new Error(`unexpected limit16 result: ${JSON.stringify(limit16)}`);
        }

        const limit8 = checkNemotronWebGpuLimits({ maxStorageBuffersPerShaderStage: 8 });
        if (limit8.ok !== true || limit8.variant.file !== 'encoder_fp16_concat8.onnx') {
          throw new Error(`unexpected limit8 result: ${JSON.stringify(limit8)}`);
        }

        const tooLow = checkNemotronWebGpuLimits({ maxStorageBuffersPerShaderStage: 7 });
        if (tooLow.ok !== false || !tooLow.reason.includes('7') || !tooLow.reason.includes('8')) {
          throw new Error(`unexpected too-low result: ${JSON.stringify(tooLow)}`);
        }

        const url = nemotronModelAssetUrl('/static/nemotron/models', 'vocab.json');
        if (url !== '/static/nemotron/models/vocab.json?v=2') {
          throw new Error(`unexpected model asset URL: ${url}`);
        }
    """
    subprocess.run(
        ["node", "--input-type=module", "-e", script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_nemotron_template_exposes_local_engine_progress_status():
    html = (ROOT / "app/templates/index.html").read_text()

    assert 'id="localEngineProgress"' in html
    assert 'function showLocalEngineProgress' in html
    assert "handleLocalEngineStatus(status)" in html
    assert "nemotron-engine.mjs?v=3" in html


def test_nemotron_template_exposes_webgpu_debug_panel():
    html = (ROOT / "app/templates/index.html").read_text()

    assert 'id="nemotronDevice"' in html
    assert 'id="checkNemotronWebGpu"' in html
    assert 'id="nemotronWebGpuDebugOutput"' in html
    assert "function checkNemotronWebGpuDebug" in html
    assert "maxStorageBuffersPerShaderStage" in html
    assert "chrome://flags/#enable-unsafe-webgpu" in html
