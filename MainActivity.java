package com.example.skincancerdetector;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.net.Uri;
import android.os.Bundle;
import android.provider.MediaStore;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AlertDialog;
import androidx.appcompat.app.AppCompatActivity;

import com.example.skincancerdetector.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class MainActivity extends AppCompatActivity {
    Button camera, gallery;
    ImageView imageView;
    TextView result;
    int imageSize = 224; // Ensure this matches the model input shape

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        camera = findViewById(R.id.button);
        gallery = findViewById(R.id.button2);
        result = findViewById(R.id.result);
        imageView = findViewById(R.id.imageView);

        Toast.makeText(this, "App Started Successfully!", Toast.LENGTH_SHORT).show();
        Log.d("APP_DEBUG", "App initialized correctly");

        camera.setOnClickListener(view -> {
            if (checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED) {
                Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                startActivityForResult(cameraIntent, 3);
            } else {
                requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
            }
        });

        gallery.setOnClickListener(view -> {
            if (checkSelfPermission(Manifest.permission.READ_EXTERNAL_STORAGE) == PackageManager.PERMISSION_GRANTED) {
                Intent galleryIntent = new Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
                startActivityForResult(galleryIntent, 1);
            } else {
                requestPermissions(new String[]{Manifest.permission.READ_EXTERNAL_STORAGE}, 101);
            }
        });
    }

    private void classifyImage(Bitmap image) {
        try {
            Log.d("APP_DEBUG", "Starting image classification...");

            Model model = Model.newInstance(getApplicationContext());

            // Ensure correct model input shape
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 224, 224, 3}, DataType.FLOAT32);
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * imageSize * imageSize * 3);
            byteBuffer.order(ByteOrder.nativeOrder());

            // Convert image pixels to Float32 array
            int[] intValues = new int[imageSize * imageSize];
            image.getPixels(intValues, 0, image.getWidth(), 0, 0, image.getWidth(), image.getHeight());

            int pixel = 0;
            for (int i = 0; i < imageSize; i++) {
                for (int j = 0; j < imageSize; j++) {
                    int val = intValues[pixel++];
                    byteBuffer.putFloat(Color.red(val) / 255.0f);
                    byteBuffer.putFloat(Color.green(val) / 255.0f);
                    byteBuffer.putFloat(Color.blue(val) / 255.0f);
                }
            }

            inputFeature0.loadBuffer(byteBuffer);
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            float[] confidences = outputFeature0.getFloatArray();

            // Log confidence values
            Log.d("APP_DEBUG", "Model Output Array Length: " + confidences.length);
            for (int i = 0; i < confidences.length; i++) {
                Log.d("APP_DEBUG", "Class " + i + " Confidence: " + confidences[i]);
            }

            // Determine class label and set color accordingly
            String resultText;
            int textColor;

            if (confidences.length == 1) {
                float malignantConfidence = confidences[0];
                if (malignantConfidence > 0.5) {
                    resultText = "Malignant (" + String.format("%.2f", malignantConfidence * 100) + "% confidence)";
                    textColor = Color.RED;
                } else {
                    resultText = "Benign (" + String.format("%.2f", (1 - malignantConfidence) * 100) + "% confidence)";
                    textColor = Color.BLACK;
                }
            } else {
                int predictedClass = confidences[0] > confidences[1] ? 0 : 1;
                resultText = (predictedClass == 0 ? "Benign" : "Malignant") +
                        " (" + String.format("%.2f", confidences[predictedClass] * 100) + "% confidence)";
                textColor = (predictedClass == 1) ? Color.RED : Color.BLACK;
            }

            // Update UI
            result.setText(resultText);
            result.setTextColor(textColor); // Set the text color
            Toast.makeText(this, resultText, Toast.LENGTH_LONG).show();
            Log.d("APP_DEBUG", "Classification completed: " + resultText);

            model.close();
        } catch (IOException e) {
            showErrorDialog("Error loading model: " + e.getMessage());
            Log.e("ModelError", "Error loading model", e);
        }
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (resultCode == RESULT_OK && data != null) {
            try {
                Bitmap image = null;
                if (requestCode == 3) { // Camera
                    image = (Bitmap) data.getExtras().get("data");
                    Log.d("APP_DEBUG", "Camera image captured.");
                } else if (requestCode == 1) { // Gallery
                    Uri imageUri = data.getData();
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), imageUri);
                    Log.d("APP_DEBUG", "Gallery image selected.");
                }

                if (image != null) {
                    image = processImage(image);
                    imageView.setImageBitmap(image);
                    classifyImage(image);
                } else {
                    showErrorDialog("Failed to retrieve image!");
                    Log.e("ImageError", "Failed to retrieve image from intent.");
                }
            } catch (Exception e) {
                showErrorDialog("Error processing image: " + e.getMessage());
                Log.e("ImageError", "Error processing image", e);
            }
        }
    }

    private Bitmap processImage(Bitmap image) {
        // Ensure correct color format
        Bitmap resizedBitmap = Bitmap.createScaledBitmap(image, imageSize, imageSize, false);

        if (resizedBitmap.getConfig() != Bitmap.Config.ARGB_8888) {
            resizedBitmap = resizedBitmap.copy(Bitmap.Config.ARGB_8888, true);
        }
        return resizedBitmap;
    }

    private void showErrorDialog(String message) {
        new AlertDialog.Builder(this)
                .setTitle("Error")
                .setMessage(message)
                .setPositiveButton("OK", (dialog, which) -> dialog.dismiss())
                .show();
    }
}
