package com.manuscript.imagegenerator

import android.content.ContentValues
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.Toast
import androidx.activity.result.IntentSenderRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.lifecycle.lifecycleScope
import com.bumptech.glide.Glide
import com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions
import com.google.mlkit.vision.documentscanner.GmsDocumentScanning
import com.google.mlkit.vision.documentscanner.GmsDocumentScanningResult
import com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions.RESULT_FORMAT_JPEG
import com.google.mlkit.vision.documentscanner.GmsDocumentScannerOptions.SCANNER_MODE_FULL
import com.manuscript.imagegenerator.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.IOException

class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var inferenceEngine: ManuscriptInferenceEngine

    private var selectedImageUri: Uri? = null
    private var inputBitmap: Bitmap? = null
    private var outputBitmap: Bitmap? = null

    private var isModelInitialized = false

    // Unified Scanner Launcher (Handles both Camera & Gallery Import)
    private val scannerLauncher = registerForActivityResult(
        ActivityResultContracts.StartIntentSenderForResult()
    ) { result ->
        if (result.resultCode == RESULT_OK) {
            val scanningResult = GmsDocumentScanningResult.fromActivityResultIntent(result.data)

            // The scanner returns cropped, cleaned pages. We take the first one.
            scanningResult?.pages?.let { pages ->
                if (pages.isNotEmpty()) {
                    val scannedUri = pages[0].imageUri
                    handleImageSelected(scannedUri)
                }
            }
        } else {
            // Scan cancelled
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        setupUI()
        initializeModel()
    }

    private fun setupUI() {
        // 1. Scan Button (Camera)
        binding.btnScan.setOnClickListener {
            startDocumentScan()
        }

        // 2. Select Button (Gallery)
        // We route this through the Scanner too, so ML Kit can auto-crop the gallery image.
        binding.btnSelectImage.setOnClickListener {
            Toast.makeText(this, "Tap the 'Import' icon in the scanner to pick from Gallery", Toast.LENGTH_LONG).show()
            startDocumentScan()
        }

        // 3. Process Button
        binding.btnProcess.setOnClickListener {
            processImage()
        }

        // 4. Save Button
        binding.btnSave.setOnClickListener {
            saveResultImage()
        }

        // 5. Clear Button
        binding.btnClear.setOnClickListener {
            clearAll()
        }

        // Initially hide process/save buttons
        updateUIState(hasImage = false, hasResult = false)
    }

    private fun initializeModel() {
        showLoading(true, "Loading AI model...")

        lifecycleScope.launch {
            try {
                inferenceEngine = ManuscriptInferenceEngine(this@MainActivity)
                val success = inferenceEngine.initialize()

                withContext(Dispatchers.Main) {
                    showLoading(false)
                    if (success) {
                        isModelInitialized = true
                        showToast("✅ Model loaded successfully!")
                        binding.tvStatus.text = "Ready to process manuscripts"
                    } else {
                        showToast("❌ Failed to load model")
                        binding.tvStatus.text = "Error: Model failed to load"
                    }
                }
            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    showToast("❌ Error: ${e.message}")
                    binding.tvStatus.text = "Error loading model"
                }
            }
        }
    }

    /**
     * Launch Google ML Kit Document Scanner
     * This handles Auto-Cropping, Edge Detection, and Cleaning.
     */
    private fun startDocumentScan() {
        val options = GmsDocumentScannerOptions.Builder()
            .setGalleryImportAllowed(true) // Allows importing existing photos for auto-cropping
            .setPageLimit(1) // We only need 1 page for restoration
            .setResultFormats(RESULT_FORMAT_JPEG)
            .setScannerMode(SCANNER_MODE_FULL)
            .build()

        val scanner = GmsDocumentScanning.getClient(options)

        scanner.getStartScanIntent(this)
            .addOnSuccessListener { intentSender ->
                scannerLauncher.launch(
                    IntentSenderRequest.Builder(intentSender).build()
                )
            }
            .addOnFailureListener { e ->
                showToast("❌ Scanner failed: ${e.message}")
                e.printStackTrace()
            }
    }

    private fun handleImageSelected(uri: Uri) {
        selectedImageUri = uri

        lifecycleScope.launch {
            try {
                // Load bitmap from URI
                inputBitmap = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { inputStream ->
                        BitmapFactory.decodeStream(inputStream)
                    }
                }

                inputBitmap?.let { bitmap ->
                    // Display input image
                    Glide.with(this@MainActivity)
                        .load(bitmap)
                        .into(binding.ivInputImage)

                    // Update UI
                    binding.tvInputInfo.text = "Input: ${bitmap.width} × ${bitmap.height} px"
                    binding.tvStatus.text = "Image loaded. Click 'Restore' to begin."

                    // FIX: Hide the placeholder text when image loads
                    binding.tvInputPlaceholder.visibility = View.GONE

                    updateUIState(hasImage = true, hasResult = false)

                    // Clear previous output
                    binding.ivOutputImage.setImageDrawable(null)
                    binding.tvOutputInfo.text = "Output: -"
                    outputBitmap = null

                    // FIX: Show the output placeholder again since we cleared the result
                    binding.tvOutputPlaceholder.visibility = View.VISIBLE

                } ?: run {
                    showToast("Failed to load image")
                }

            } catch (e: Exception) {
                showToast("Error loading image: ${e.message}")
                e.printStackTrace()
            }
        }
    }

    private fun processImage() {
        if (!isModelInitialized) {
            showToast("Model not initialized yet. Please wait.")
            return
        }

        val bitmap = inputBitmap ?: run {
            showToast("Please select an image first")
            return
        }

        showLoading(true, "Restoring manuscript...")
        binding.tvStatus.text = "AI is restoring your manuscript..."

        lifecycleScope.launch {
            try {
                val startTime = System.currentTimeMillis()

                // Run inference
                val result = inferenceEngine.processImage(bitmap)

                val processingTime = System.currentTimeMillis() - startTime

                withContext(Dispatchers.Main) {
                    showLoading(false)

                    if (result.isSuccess) {
                        outputBitmap = result.getOrNull()

                        outputBitmap?.let { output ->
                            // Display output image
                            Glide.with(this@MainActivity)
                                .load(output)
                                .into(binding.ivOutputImage)

                            // Update info
                            binding.tvOutputInfo.text =
                                "Output: ${output.width} × ${output.height} px"
                            binding.tvStatus.text =
                                "✅ Processing complete in ${processingTime}ms"

                            // FIX: Hide the output placeholder text
                            binding.tvOutputPlaceholder.visibility = View.GONE

                            updateUIState(hasImage = true, hasResult = true)
                            showToast("✅ Manuscript restored successfully!")
                        }
                    } else {
                        val error = result.exceptionOrNull()
                        binding.tvStatus.text = "❌ Processing failed: ${error?.message}"
                        showToast("Processing failed")
                    }
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showLoading(false)
                    binding.tvStatus.text = "❌ Error: ${e.message}"
                    showToast("Error during processing")
                    e.printStackTrace()
                }
            }
        }
    }

    private fun saveResultImage() {
        val bitmap = outputBitmap ?: run {
            showToast("No result to save. Process an image first.")
            return
        }

        lifecycleScope.launch {
            try {
                val saved = withContext(Dispatchers.IO) {
                    saveBitmapToGallery(bitmap)
                }

                withContext(Dispatchers.Main) {
                    if (saved) {
                        showToast("✅ Image saved to Gallery!")
                        binding.tvStatus.text = "Image saved successfully"
                    } else {
                        showToast("❌ Failed to save image")
                    }
                }

            } catch (e: Exception) {
                withContext(Dispatchers.Main) {
                    showToast("Error saving: ${e.message}")
                    e.printStackTrace()
                }
            }
        }
    }

    private fun saveBitmapToGallery(bitmap: Bitmap): Boolean {
        return try {
            val filename = "manuscript_restored_${System.currentTimeMillis()}.png"

            val contentValues = ContentValues().apply {
                put(MediaStore.Images.Media.DISPLAY_NAME, filename)
                put(MediaStore.Images.Media.MIME_TYPE, "image/png")
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/ManuscriptRegenerator")
                    put(MediaStore.Images.Media.IS_PENDING, 1)
                }
            }

            val uri = contentResolver.insert(
                MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
                contentValues
            )

            uri?.let {
                contentResolver.openOutputStream(it)?.use { outputStream ->
                    bitmap.compress(Bitmap.CompressFormat.PNG, 100, outputStream)
                }

                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
                    contentValues.clear()
                    contentValues.put(MediaStore.Images.Media.IS_PENDING, 0)
                    contentResolver.update(it, contentValues, null, null)
                }
                true
            } ?: false

        } catch (e: IOException) {
            e.printStackTrace()
            false
        }
    }

    private fun clearAll() {
        inputBitmap = null
        outputBitmap = null
        selectedImageUri = null

        binding.ivInputImage.setImageDrawable(null)
        binding.ivOutputImage.setImageDrawable(null)
        binding.tvInputInfo.text = "Input: -"
        binding.tvOutputInfo.text = "Output: -"
        binding.tvStatus.text = "Ready to process manuscripts"

        // FIX: Show both placeholders again
        binding.tvInputPlaceholder.visibility = View.VISIBLE
        binding.tvOutputPlaceholder.visibility = View.VISIBLE

        updateUIState(hasImage = false, hasResult = false)
    }

    private fun updateUIState(hasImage: Boolean, hasResult: Boolean) {
        binding.btnProcess.isEnabled = hasImage && isModelInitialized
        binding.btnSave.isEnabled = hasResult
        binding.btnClear.isEnabled = hasImage || hasResult

        // Scan/Select should usually stay enabled so user can change image
        binding.btnScan.isEnabled = true
        binding.btnSelectImage.isEnabled = true

        // Update button visibility
        binding.btnProcess.alpha = if (hasImage && isModelInitialized) 1.0f else 0.5f
        binding.btnSave.alpha = if (hasResult) 1.0f else 0.5f
    }

    private fun showLoading(show: Boolean, message: String = "Loading...") {
        binding.progressBar.visibility = if (show) View.VISIBLE else View.GONE
        binding.tvLoadingMessage.visibility = if (show) View.VISIBLE else View.GONE
        binding.tvLoadingMessage.text = message

        // Disable buttons while loading
        binding.btnScan.isEnabled = !show
        binding.btnSelectImage.isEnabled = !show
        binding.btnProcess.isEnabled = !show && inputBitmap != null
        binding.btnSave.isEnabled = !show && outputBitmap != null
        binding.btnClear.isEnabled = !show
    }

    private fun showToast(message: String) {
        Toast.makeText(this, message, Toast.LENGTH_SHORT).show()
    }

    override fun onDestroy() {
        super.onDestroy()
        if (::inferenceEngine.isInitialized) {
            inferenceEngine.close()
        }
        inputBitmap?.recycle()
        outputBitmap?.recycle()
    }
}