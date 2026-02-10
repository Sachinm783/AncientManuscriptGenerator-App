package com.manuscript.imagegenerator

import android.content.Context
import android.graphics.Bitmap
import ai.onnxruntime.*
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.util.*

/**
 * ONNX Inference Engine for RGB Manuscript Regeneration
 *
 * This class handles:
 * - Loading the ONNX model from assets
 * - Preprocessing RGB images to model input format
 * - Running inference
 * - Postprocessing output back to RGB Bitmap
 */
class ManuscriptInferenceEngine(private val context: Context) {

    private var ortSession: OrtSession? = null
    private var ortEnvironment: OrtEnvironment? = null

    companion object {
        private const val MODEL_NAME = "manuscript_model.onnx"
        private const val INPUT_NAME = "input"
        private const val OUTPUT_NAME = "output"

        // Model expects RGB images (3 channels)
        private const val NUM_CHANNELS = 3

        // Model can handle any size (dynamic axes), but we'll process at reasonable sizes
        private const val MAX_DIMENSION = 1024
    }

    /**
     * Initialize the ONNX Runtime and load the model
     */
    suspend fun initialize(): Boolean = withContext(Dispatchers.IO) {
        try {
            // Create ONNX environment
            ortEnvironment = OrtEnvironment.getEnvironment()

            // Load model from assets
            val modelBytes = context.assets.open(MODEL_NAME).use { it.readBytes() }

            // Create session options
            val sessionOptions = OrtSession.SessionOptions().apply {
                // Optimize for mobile
                setIntraOpNumThreads(4)
                setInterOpNumThreads(4)
                setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            }

            // Create session
            ortSession = ortEnvironment?.createSession(modelBytes, sessionOptions)

            println("✅ ONNX Model loaded successfully")
            println("   Model: $MODEL_NAME")
            println("   Input: RGB [batch, 3, height, width]")
            println("   Output: RGB [batch, 3, height, width]")

            true
        } catch (e: Exception) {
            println("❌ Failed to load ONNX model: ${e.message}")
            e.printStackTrace()
            false
        }
    }

    /**
     * Process a manuscript image
     *
     * @param inputBitmap The degraded manuscript image (RGB)
     * @return The restored manuscript image (RGB)
     */
    suspend fun processImage(inputBitmap: Bitmap): Result<Bitmap> = withContext(Dispatchers.IO) {
        try {
            val session = ortSession ?: return@withContext Result.failure(
                Exception("Model not initialized. Call initialize() first.")
            )

            // Step 1: Preprocess - Convert Bitmap to model input format
            val (inputTensor, originalWidth, originalHeight) = preprocessImage(inputBitmap)

            // Step 2: Run inference
            val startTime = System.currentTimeMillis()
            val outputs = session.run(Collections.singletonMap(INPUT_NAME, inputTensor))
            val inferenceTime = System.currentTimeMillis() - startTime

            // Step 3: Postprocess - Convert output back to Bitmap
            val outputTensor = outputs[0].value as Array<Array<Array<FloatArray>>>
            val resultBitmap = postprocessImage(outputTensor, originalWidth, originalHeight)

            outputs.close()
            inputTensor.close()

            println("✅ Inference complete in ${inferenceTime}ms")
            Result.success(resultBitmap)

        } catch (e: Exception) {
            println("❌ Inference failed: ${e.message}")
            e.printStackTrace()
            Result.failure(e)
        }
    }

    /**
     * Preprocess: Bitmap (RGB) → FloatArray [1, 3, H, W] normalized to [0, 1]
     */
    /**
     * Preprocess: Bitmap (RGB) → FloatArray [1, 3, H, W] normalized to [0, 1]
     * OPTIMIZED: Ensures dimensions are multiples of 32 to prevent U-Net Concat errors.
     */
    private fun preprocessImage(bitmap: Bitmap): Triple<OnnxTensor, Int, Int> {
        val originalWidth = bitmap.width
        val originalHeight = bitmap.height

        // U-Net stride requirement (5 pooling layers = 2^5 = 32)
        val stride = 32

        // 1. Calculate target dimensions (respecting MAX_DIMENSION and stride)
        var targetWidth = originalWidth
        var targetHeight = originalHeight

        // Scale down if exceeds max dimension
        if (originalWidth > MAX_DIMENSION || originalHeight > MAX_DIMENSION) {
            val scale = MAX_DIMENSION.toFloat() / maxOf(originalWidth, originalHeight)
            targetWidth = (originalWidth * scale).toInt()
            targetHeight = (originalHeight * scale).toInt()
        }

        // 2. Round UP to nearest multiple of stride (better than rounding down)
        // This preserves more image detail
        targetWidth = ((targetWidth + stride - 1) / stride) * stride
        targetHeight = ((targetHeight + stride - 1) / stride) * stride

        // 3. Ensure minimum dimensions (at least one stride)
        targetWidth = maxOf(targetWidth, stride)
        targetHeight = maxOf(targetHeight, stride)

        // 4. Create resized bitmap if needed
        val resizedBitmap = if (targetWidth != originalWidth || targetHeight != originalHeight) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        } else {
            bitmap
        }

        val width = resizedBitmap.width
        val height = resizedBitmap.height

        println("📐 Image dimensions: ${originalWidth}×${originalHeight} → ${width}×${height} (stride: ${stride})")

        // 5. Extract RGB pixels
        val pixels = IntArray(width * height)
        resizedBitmap.getPixels(pixels, 0, width, 0, 0, width, height)

        // 6. Convert to [1, 3, H, W] float array normalized to [0, 1]
        val floatArray = FloatArray(NUM_CHANNELS * height * width)

        var idx = 0
        // Channel-first order: [R channel, G channel, B channel]
        for (c in 0 until NUM_CHANNELS) {
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val pixel = pixels[y * width + x]
                    val value = when (c) {
                        0 -> ((pixel shr 16) and 0xFF) / 255f  // R
                        1 -> ((pixel shr 8) and 0xFF) / 255f   // G
                        2 -> (pixel and 0xFF) / 255f           // B
                        else -> 0f
                    }
                    floatArray[idx++] = value
                }
            }
        }

        // 7. Create ONNX tensor
        val shape = longArrayOf(1, NUM_CHANNELS.toLong(), height.toLong(), width.toLong())
        val tensor = OnnxTensor.createTensor(
            ortEnvironment,
            FloatBuffer.wrap(floatArray),
            shape
        )

        // 8. Clean up temporary bitmap if we created one
        if (resizedBitmap !== bitmap) {
            resizedBitmap.recycle()
        }

        // Return tensor and ORIGINAL dimensions for proper output scaling
        return Triple(tensor, originalWidth, originalHeight)
    }

    /**
     * Postprocess: FloatArray [1, 3, H, W] → Bitmap (RGB)
     */
    private fun postprocessImage(
        outputArray: Array<Array<Array<FloatArray>>>,
        targetWidth: Int,
        targetHeight: Int
    ): Bitmap {
        // Output shape: [1, 3, H, W]
        val channels = outputArray[0]
        val height = channels[0].size
        val width = channels[0][0].size

        // Create pixel array
        val pixels = IntArray(width * height)

        for (y in 0 until height) {
            for (x in 0 until width) {
                // Get RGB values (already in [0, 1] range due to sigmoid)
                val r = (channels[0][y][x] * 255f).coerceIn(0f, 255f).toInt()
                val g = (channels[1][y][x] * 255f).coerceIn(0f, 255f).toInt()
                val b = (channels[2][y][x] * 255f).coerceIn(0f, 255f).toInt()

                // Combine into ARGB pixel
                pixels[y * width + x] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
            }
        }

        // Create bitmap
        val bitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        bitmap.setPixels(pixels, 0, width, 0, 0, width, height)

        // Resize back to original dimensions if needed
        return if (width != targetWidth || height != targetHeight) {
            Bitmap.createScaledBitmap(bitmap, targetWidth, targetHeight, true)
        } else {
            bitmap
        }
    }

    /**
     * Get model info for display
     */
    fun getModelInfo(): String {
        val session = ortSession ?: return "Model not loaded"

        return buildString {
            appendLine("📦 Model Information:")
            appendLine("  Name: $MODEL_NAME")
            appendLine("  Type: RGB U-Net (3→3 channels)")
            appendLine("  Input: RGB image [batch, 3, H, W]")
            appendLine("  Output: RGB image [batch, 3, H, W]")
            appendLine("  Purpose: Manuscript restoration")
            appendLine("  Dynamic size: Yes (any dimensions)")
        }
    }

    /**
     * Clean up resources
     */
    fun close() {
        ortSession?.close()
        ortEnvironment?.close()
        ortSession = null
        ortEnvironment = null
        println("🧹 ONNX resources cleaned up")
    }
}

/**
 * Data class to hold processing results with metadata
 */
data class ProcessingResult(
    val bitmap: Bitmap,
    val inferenceTimeMs: Long,
    val inputSize: Pair<Int, Int>,
    val outputSize: Pair<Int, Int>
)