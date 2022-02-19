package edu.asu.ame.meteor.speedytiltshift2022;

import android.graphics.Bitmap;
import android.util.Log;



public class SpeedyTiltShift {

    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }


    public static int get_p_val(int x, int y, int[] pixels, int width, int height) {
        int res = 0;
        if(y < height && y >= 0 && x >= 0 && x < width) res = pixels[y*width+x];
        return res;

    }


    public static float[] compute_gaussian_blur_weight_vector(int kernel_radius, float sigma) {
        kernel_radius=500;
        float[] weight_vector = new float[kernel_radius + 1];
        for (int i = 0; i <= kernel_radius ; i++) {
            // G(k) = 𝟏/*𝟐𝝅𝝈𝟐 𝒆𝒙𝒑(− 𝒌𝟐/𝟐𝝈𝟐)

            float two_sigma_square = (float) (2.0 * sigma * sigma);
            float part_a = (float) ((float) (1.00) / (Math.sqrt(two_sigma_square * Math.PI)));
            float part_b = (float) Math.exp((-i * i)/two_sigma_square);

            weight_vector[i] = part_a * part_b;
        }

        return weight_vector;
    }


    public static int compute_gaussian_blur_weight_vector_matrix(int r, int y, int x, int[] pixels, int width, float sigma, int height, int iteration) {
        float[] weight_vector = compute_gaussian_blur_weight_vector(r, sigma);
        int pt_r = 0, pt_g = 0, pt_b = 0, pt_a = 0;

        if(iteration == 1) {
            for (int i = -r; i <= r; i++) {
                float gaussian_vector_val = weight_vector[Math.abs(i)];
                int p = get_p_val(x, y+i, pixels, width, height);
                int A = (p >> 24) & 0xff,  R = (p >> 16) & 0xff, G = (p >> 8) & 0xff, B = p & 0xff;

                pt_r += R * gaussian_vector_val;
                pt_g += G * gaussian_vector_val;
                pt_b += B * gaussian_vector_val;
                pt_a += A * gaussian_vector_val;

            }
        } else {
            for (int i = -r; i <= r; i++) {
                float gaussian_vector_val = weight_vector[Math.abs(i)];
                int p = get_p_val(x+i, y, pixels, width, height);
                int B = p & 0xff, G = (p >> 8) & 0xff, R = (p >> 16) & 0xff, A = (p >> 24) & 0xff;

                pt_r += R * gaussian_vector_val;
                pt_g += G * gaussian_vector_val;
                pt_b += B * gaussian_vector_val;
                pt_a += A * gaussian_vector_val;
            }
        }
        int int_pt_a = (int) pt_a;
        int int_pt_r = (int) pt_r;
        int int_pt_g = (int) pt_g;
        int int_pt_b = (int) pt_b;

        return (int) (int_pt_a & 0xff) << 24 | (int_pt_r & 0xff) << 16 | (int_pt_g & 0xff) << 8 | (int_pt_b & 0xff);

    }


    public static Bitmap tiltshift_java(Bitmap input, float s_far, float s_near, int a0, int a1, int a2, int a3){
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);

        int width=input.getWidth();
        int height=input.getHeight();

        int r_far =(int) Math.ceil(3*s_far);
        int r_near = (int) Math.ceil(3*s_near);
        int[] pixels = new int[width*height];
        input.getPixels(pixels,0, width,0,0,width,height);

        for(int i = 1; i <= 2; i++ ) {
            int[] pixels1 = pixels;

            for (int y = 0; y < height; y++) {
                for (int x = 0; x < width; x++) {

                    if (y < a0) {
                        if (s_far >= 0.6)
                            pixels[y * width + x] = compute_gaussian_blur_weight_vector_matrix(r_far, y, x, pixels1, width, s_far, height, i);

                    } else if ( y < a1) {
                        float sigma = s_far * (a1 - y) / (a1 - a0);
                        if (sigma >= 0.6)
                            pixels[y * width + x] = compute_gaussian_blur_weight_vector_matrix(r_far, y, x, pixels1, width, sigma, height, i);

                    } else if ( y < a2) {
                        Log.d("NO BLUR", "no blur region");

                    } else if (y < a3) {
                        float sigma = s_near * (y - a2) / (a3 - a2);
                        if (sigma >= 0.6)
                            pixels[y * width + x] = compute_gaussian_blur_weight_vector_matrix(r_near, y, x, pixels1, width, sigma, height, i);

                    } else {
                        if (s_near >= 0.6)
                            pixels[y * width + x] = compute_gaussian_blur_weight_vector_matrix(r_near, y, x, pixels1, width, s_near, height, i);

                    }
                }
            }
        }

        outBmp.setPixels(pixels,0, width,0,0,width,height);
        return outBmp;

    }


    public static Bitmap tiltshift_cpp(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());

        tiltshiftcppnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);

        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        return outBmp;
    }
    public static Bitmap tiltshift_neon(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());

        tiltshiftneonnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);

        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        return outBmp;
    }


    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public static native int tiltshiftcppnative(int[] inputPixels, int[] outputPixels, int width, int height, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3);
    public static native int tiltshiftneonnative(int[] inputPixels, int[] outputPixels, int width, int height, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3);

}
