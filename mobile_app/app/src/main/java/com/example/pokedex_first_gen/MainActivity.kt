package com.example.pokedex_first_gen

import android.Manifest
import android.animation.ObjectAnimator
import android.animation.ValueAnimator
import android.app.Activity
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.view.animation.DecelerateInterpolator
import android.widget.*
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.Tensor
import java.io.File
import java.io.FileOutputStream

class MainActivity : AppCompatActivity() {

    private lateinit var module: Module

    private lateinit var labelsMap: Map<String, String>
    private lateinit var labelsList: List<String>

    private lateinit var imageView: ImageView
    private lateinit var resultTextView: TextView
    private lateinit var cameraButton: Button

    private lateinit var pokeImages: List<ImageView>
    private lateinit var pokeNames: List<TextView>
    private lateinit var pokeBars: List<ProgressBar>

    private lateinit var topContainers: List<View>

    private val CAMERA_REQUEST = 100
    private val CAMERA_PERMISSION_CODE = 200

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        bindViews()
        setupModel()
        setupLabels()

        imageView.scaleX = -1f

        setResultsVisible(false)

        cameraButton.setOnClickListener {
            requestCamera()
        }
    }

    // ---------------- UI BIND ----------------
    private fun bindViews() {

        imageView = findViewById(R.id.imageView)
        resultTextView = findViewById(R.id.resultTextView)
        cameraButton = findViewById(R.id.cameraButton)

        pokeImages = listOf(
            findViewById(R.id.imageTop1),
            findViewById(R.id.imageTop2),
            findViewById(R.id.imageTop3)
        )

        pokeNames = listOf(
            findViewById(R.id.nameTop1),
            findViewById(R.id.nameTop2),
            findViewById(R.id.nameTop3)
        )

        pokeBars = listOf(
            findViewById(R.id.barTop1),
            findViewById(R.id.barTop2),
            findViewById(R.id.barTop3)
        )

        topContainers = listOf(
            findViewById(R.id.top1Container),
            findViewById(R.id.top2Container),
            findViewById(R.id.top3Container)
        )
    }

    // ---------------- MODEL ----------------
    private fun setupModel() {
        module = Module.load(assetFilePath(this, "model_mobile.pt"))
    }

    // ---------------- LABELS ----------------
    private fun setupLabels() {
        val tempMap = mutableMapOf<String, String>()
        val tempList = mutableListOf<String>()

        assets.open("labels.txt").bufferedReader().forEachLine {
            val parts = it.split(",")
            tempMap[parts[0]] = parts[1]
            tempList.add(parts[0])
        }

        labelsMap = tempMap
        labelsList = tempList
    }

    // ---------------- CAMERA ----------------
    private fun requestCamera() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA)
            != PackageManager.PERMISSION_GRANTED
        ) {
            ActivityCompat.requestPermissions(
                this,
                arrayOf(Manifest.permission.CAMERA),
                CAMERA_PERMISSION_CODE
            )
        } else {
            openCamera()
        }
    }

    private fun openCamera() {
        val intent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(intent, CAMERA_REQUEST)
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (requestCode == CAMERA_REQUEST && resultCode == Activity.RESULT_OK) {
            val bitmap = data?.extras?.get("data") as Bitmap

            imageView.setImageBitmap(bitmap)

            setResultsVisible(true)
            predict(bitmap)
        }
    }

    // ---------------- PREDICTION ----------------
    private fun predict(bitmap: Bitmap) {

        val scaled = Bitmap.createScaledBitmap(bitmap, 224, 224, true)

        val input = FloatArray(3 * 224 * 224)
        var r = 0
        var g = 224 * 224
        var b = 2 * 224 * 224

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = scaled.getPixel(x, y)

                input[r++] = ((pixel shr 16 and 0xFF) / 255f - 0.485f) / 0.229f
                input[g++] = ((pixel shr 8 and 0xFF) / 255f - 0.456f) / 0.224f
                input[b++] = ((pixel and 0xFF) / 255f - 0.406f) / 0.225f
            }
        }

        val tensor = Tensor.fromBlob(input, longArrayOf(1, 3, 224, 224))
        val output = module.forward(IValue.from(tensor)).toTensor()
        val scores = output.dataAsFloatArray

        val probs = softmax(scores)

        val top3 = probs.mapIndexed { i, v -> i to v }
            .sortedByDescending { it.second }
            .take(3)

        for ((i, pair) in top3.withIndex()) {

            val nameEn = labelsList[pair.first]
            val nameFr = labelsMap[nameEn] ?: nameEn
            val prob = (pair.second * 100).toInt()

            val formattedNameFr = nameFr.replaceFirstChar { it.uppercase() }
            pokeNames[i].text = "$formattedNameFr ($prob%)"

            val resId = getImageResourceName(nameEn)
            if (resId != 0) pokeImages[i].setImageResource(resId)

            animateBar(pokeBars[i], prob)
            animateAppear(topContainers[i])

            if (i == 0) {
                applyGlow(topContainers[i])
            } else {
                removeGlow(topContainers[i])
            }
        }

        resultTextView.text = "Voici ton reflet !"
    }

    // ---------------- ANIMATIONS ----------------
    private fun animateAppear(view: View) {
        view.alpha = 0f
        view.translationY = 50f
        view.scaleX = 0.95f
        view.scaleY = 0.95f

        view.animate()
            .alpha(1f)
            .translationY(0f)
            .scaleX(1f)
            .scaleY(1f)
            .setDuration(450)
            .setInterpolator(DecelerateInterpolator())
            .start()
    }

    private fun animateBar(bar: ProgressBar, value: Int) {
        val anim = ObjectAnimator.ofInt(bar, "progress", 0, value)
        anim.duration = 900
        anim.interpolator = DecelerateInterpolator()
        anim.start()
    }

    // ---------------- GLOW ----------------
    private fun applyGlow(view: View) {
        view.setBackgroundResource(R.drawable.glow_bg)
    }

    private fun removeGlow(view: View) {
        view.setBackgroundResource(R.drawable.pokemon_card)
    }

    // ---------------- UX ----------------
    private fun setResultsVisible(visible: Boolean) {
        val alpha = if (visible) 1f else 0f

        for (v in topContainers) {
            v.alpha = alpha
        }
    }

    // ---------------- IMAGE RES ----------------
    private fun getImageResourceName(name: String): Int {
        val formatted = name.lowercase().replace("-", "_")
        return resources.getIdentifier(formatted, "drawable", packageName)
    }

    // ---------------- SOFTMAX ----------------
    private fun softmax(x: FloatArray): FloatArray {
        val max = x.maxOrNull() ?: 0f
        val exp = x.map { Math.exp((it - max).toDouble()) }
        val sum = exp.sum()
        return exp.map { (it / sum).toFloat() }.toFloatArray()
    }

    // ---------------- ASSET LOADER ----------------
    private fun assetFilePath(context: Context, name: String): String {
        val file = File(context.filesDir, name)
        if (file.exists()) return file.absolutePath

        context.assets.open(name).use { input ->
            FileOutputStream(file).use { output ->
                input.copyTo(output)
            }
        }
        return file.absolutePath
    }
}