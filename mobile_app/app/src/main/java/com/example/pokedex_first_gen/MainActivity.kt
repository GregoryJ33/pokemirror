package com.example.pokedex_first_gen

import android.Manifest
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream
import java.io.InputStream

class MainActivity : AppCompatActivity() {

    private lateinit var module: Module
    private lateinit var labelsMap: Map<String, String>

    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var cameraButton: Button
    private val CAMERA_REQUEST = 100
    private val CAMERA_PERMISSION_CODE = 200

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        cameraButton = findViewById(R.id.cameraButton)

        // 1️⃣ Charger le modèle TorchScript
        module = Module.load(assetFilePath(this, "model_mobile.pt"))

        // 2️⃣ Charger labels
        labelsMap = assets.open("labels.txt").bufferedReader().useLines { lines ->
            lines.associate { line ->
                val parts = line.split(",")
                // parts[0] est l'anglais (la clé), parts[1] est le français (la valeur)
                parts[0].trim() to parts[1].trim()
            }
        }

        // 3️⃣ Bouton caméra
        cameraButton.setOnClickListener {
            if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
                ActivityCompat.requestPermissions(this, arrayOf(Manifest.permission.CAMERA), CAMERA_PERMISSION_CODE)
            } else {
                openCamera()
            }
        }


    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, CAMERA_REQUEST)
    }

    override fun onRequestPermissionsResult(requestCode: Int, permissions: Array<String>, grantResults: IntArray) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_CODE && grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            openCamera()
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {
            val bitmap = data?.extras?.get("data") as Bitmap
            imageView.setImageBitmap(bitmap)
            resultTextView.text = predict(bitmap)
        }
    }

    // 🔹 Fonction Top-3
    private fun predict(bitmap: Bitmap): String {
        val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        val input = FloatArray(3 * 224 * 224)
        var idxR = 0
        var idxG = 224 * 224
        var idxB = 2 * 224 * 224

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = scaled.getPixel(x, y)
                val r = ((pixel shr 16) and 0xFF) / 255.0f
                val g = ((pixel shr 8) and 0xFF) / 255.0f
                val b = (pixel and 0xFF) / 255.0f

                input[idxR++] = (r - mean[0]) / std[0]
                input[idxG++] = (g - mean[1]) / std[1]
                input[idxB++] = (b - mean[2]) / std[2]
            }
        }

        val tensor = Tensor.fromBlob(input, longArrayOf(1, 3, 224, 224))
        val output = module.forward(IValue.from(tensor)).toTensor()
        val scores = output.dataAsFloatArray

        val softmaxScores = softmax(scores)
        val top3 = softmaxScores
            .mapIndexed { index, score -> index to score }
            .sortedByDescending { it.second }
            .take(3)

        val result = StringBuilder("Top 3 Pokémon :\n")
        for ((index, score) in top3) {
            // 1. On récupère le nom anglais original (clé de la map)
            val englishName = labelsMap.keys.elementAt(index)

            // 2. On récupère la traduction française
            val frenchName = labelsMap[englishName] ?: englishName
            val probability = (score * 100).toInt()
            result.append("$frenchName : $probability%\n")
        }

        return result.toString()
    }

    private fun softmax(values: FloatArray): FloatArray {
        val max = values.maxOrNull() ?: 0f
        val exp = values.map { Math.exp((it - max).toDouble()) }
        val sum = exp.sum()
        return exp.map { (it / sum).toFloat() }.toFloatArray()
    }

    // 🔹 Copier le modèle depuis assets
    private fun assetFilePath(context: Context, assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) return file.absolutePath

        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }
}
