package edu.asu.ame.meteor.speedytiltshift2022;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.util.Log;
import java.io.* ;
import android.widget.TextView;
import android.os.*;
import java.lang.*;

public class SpeedyTiltShift {
    static SpeedyTiltShift Singleton = new SpeedyTiltShift();
    // Used to load the 'native-lib' library on application startup.
    static {
        System.loadLibrary("native-lib");
    }
    
    public static long TimeInterval=0;
	
    // Defining class variables by extending thread class function
	
    public static class Sigma_FN extends Thread {
        private static int picHeight,picWidth;
        private int os0,os1,os2,os3;
        private float sigma;
        private int[] pixels;
        public  static int[] pixelsOut;
        private String threadName;
        
        Sigma_FN( String name, float Sigma,int a0,int a1,int a2,int a3,int[] pixel,int imgHeight,int imgWidth) 
		{
			
			picHeight = imgHeight;
            picWidth = imgWidth;
            pixels = pixel;
            pixelsOut = pixels;
            sigma = Sigma;
			
            threadName = name;
            os0 = a0;
            os1 = a1;
            os2 = a2;
            os3 = a3;

           Log.d("TILTSHIFT_JAVA_THREAD","Creating "+ threadName +",a0:" + os0 +",a1:" + os1+",a2:" + os2+",a3:" + os3+",H:" + picHeight +",W:" + picWidth+",Sigma:"+ Sigma +",input Pixels length:" +pixels.length+",O/p Pixel:"+pixelsOut.length);
        
		}
		
        public void run()
        {
            try
            {
                if (threadName == "Thread sigma far") {
                    if(sigma >= 0.6){
					
                    int[] k= calK(sigma);  // Kernel radius is determined and indices are assigned using sigma
                    double[] kMatrix = GaussianWeights(k, sigma);  //Determine the gaussian weights  using sigma and kernel
					
                    for (int i=0; i<=os0; i++){
														//Calculate the result of convolution with Gaussian kernel for each R,G,B,A color
                        for(int j=0; j<picWidth; j++){
                            int pBB = Vector_1("BB", j, i, picWidth, k, kMatrix, pixels);
                            int pRR = Vector_1("RR", j, i, picWidth, k, kMatrix, pixels);
                            int pGG = Vector_1("GG", j, i, picWidth, k, kMatrix, pixels);
                            int pAA = Vector_1("AA", j, i, picWidth, k, kMatrix, pixels);
                            pixelsOut[i * picWidth + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                        }
                    }
                    
                }
                    else{							// sigma value less than 0.6
                        for (int i=0; i<=os0; i++){
                            for(int j=0; j<picWidth; j++){
                                pixelsOut[i * picWidth + j] = pixels[i * picWidth + j]; //pixels are stored in pixelsOut
                            }}
                    }
                }
				       
                //Region 2 - between row a0 and a1
                else if(threadName == "Thread-sigma-far-gradual") {
                    float sigmaY;
                    for (int i = os0; i < os1 ; i++) {
                       
                        sigmaY= sigma*((float) (os1-i)/(os1-os0));
                        if(sigmaY >= 0.6) {
                       
                            int[] k = calK(sigmaY);                        //Calculating radius vector
                            
                            double[] kMatrix = GaussianWeights(k, sigmaY);  //Calculate the Gaussian Matrix
                            for (int j = 0; j < picWidth; j++) {                          
                                int pBB = Vector_1("BB", j, i, picWidth, k, kMatrix, pixels);
                                int pRR = Vector_1("RR", j, i, picWidth, k, kMatrix, pixels);
                                int pGG = Vector_1("GG", j, i, picWidth, k, kMatrix, pixels);
                                int pAA = Vector_1("AA", j, i, picWidth, k, kMatrix, pixels);
           
                                pixelsOut[i * picWidth + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                            }
                        }
                      
                        else{
                            for(int z = i ; z < os1 ; z++){               
                                for (int j = 0; j < picWidth; j++) {
                                    pixelsOut[z * picWidth + j] = pixels[z * picWidth + j];
                                }
                            }
                            break;
                        }
                    }
                }
				//Region 3 - between row a1 and a2
        
                //The pixels in this region are left untouched
                else if (threadName == "Thread-sigma-no-blur") {
                    for (int i = os1; i < os2 ; i++) {                      
                        for (int j = 0; j < picWidth; j++) {       //Copy pixels from original image to the pixelsOut  
                            pixelsOut[i * picWidth + j] = pixels[i * picWidth + j];
                        }
                    }                  
                }
                //Region 4 - between row a2 and a3
                else if(threadName == "Thread-sigma-near-gradual") { float sigmaX;
                    for (int i = os3; i > os2; i--) {
                        sigmaX = sigma * ((float)(i-os2)/(os3-os2));
                        if (sigmaX >= 0.6) {
                            int[] k = calK(sigmaX);                        //Calculating radius vector                    
                            double[] kMatrix = GaussianWeights(k, sigmaX);  //Calculate the Gaussian Matrix
                            for (int j = 0; j < picWidth; j++) {                          
                                int pBB = Vector_1("BB", j, i, picWidth, k, kMatrix, pixels);
                                int pRR = Vector_1("RR", j, i, picWidth, k, kMatrix, pixels);
                                int pGG = Vector_1("GG", j, i, picWidth, k, kMatrix, pixels);
                                int pAA = Vector_1("AA", j, i, picWidth, k, kMatrix, pixels);
                                
                                pixelsOut[i * picWidth + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                            }
                        }
                        
                        else {
                          
                            for(int z = i; z>os2 ; z--) {
                                for (int j = 0; j < picWidth; j++) {
                                    pixelsOut[i * picWidth + j] = pixels[i * picWidth + j];
                                }
                            }
                            break;
                        }
                    }
                }
				
				//Region 5 - between row a3 and final
				
                else if(threadName == "Thread-sigma-near-solid"){
                    if(sigma >= 0.6){
                        int[] k= calK(sigma);
                        double[] kMatrix = GaussianWeights(k, sigma);
                        for (int i=os3; i < picHeight; i++) {
                            for (int j = 0; j < picWidth; j++) {
                                int pBB = Vector_1("BB", j, i, picWidth, k, kMatrix, pixels);
                                int pRR = Vector_1("RR", j, i, picWidth, k, kMatrix, pixels);
                                int pGG = Vector_1("GG", j, i, picWidth, k, kMatrix, pixels);
                                int pAA = Vector_1("AA", j, i, picWidth, k, kMatrix, pixels);

                                pixelsOut[i * picWidth + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                            }
                        }
                        }
                    else {
                        for (int i = os3; i <= picHeight; i++) {
                            for (int j = 0; j < picWidth; j++) {
                                pixelsOut[i * picWidth + j] = pixels[i * picWidth + j];
                            }
                        }
                    }
                   
                }
                
            }
            catch (Exception e)
            {
                System.out.println ("Exception is caught"+ threadName +e.toString());
            }
        }
    }

    public static Bitmap tiltshift_java(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        long javaStart = System.currentTimeMillis(); 
        // Calculate the width and height of the picture
        int picHeight = input.getHeight();
        int picWidth = input.getWidth();
        Bitmap outBmp = Bitmap.createBitmap(picWidth, picHeight, Bitmap.Config.ARGB_8888);
        int[] pixels = new int[picHeight*picWidth];
        input.getPixels(pixels,0,picWidth,0,0,picWidth,picHeight);

        Log.d("TILTSHIFT_JAVA","Started:"+picWidth+","+picHeight+","+pixels.length+"a0:"+a0+", a1:"+a1+", a2:"+a2+", a3:"+a3);
        // Main Class
        Sigma_FN SigmaFS = new Sigma_FN("Thread sigma far",sigma_far,a0,a1,a2,a3,pixels,picHeight,picWidth);
        Sigma_FN SigmaNS = new Sigma_FN("Thread-sigma-near-solid",sigma_near,a0,a1,a2,a3,pixels,picHeight,picWidth);
        Sigma_FN SigmaBlur = new Sigma_FN("Thread-sigma-no-blur",sigma_near,a0,a1,a2,a3,pixels,picHeight,picWidth);
        Sigma_FN SigmaFG = new Sigma_FN("Thread-sigma-far-gradual",sigma_far,a0,a1,a2,a3,pixels,picHeight,picWidth);
        Sigma_FN SigmaNG = new Sigma_FN("Thread-sigma-near-gradual",sigma_near,a0,a1,a2,a3,pixels,picHeight,picWidth);

        SigmaFS.start();
        SigmaNS.start();
        SigmaBlur.start();
        SigmaFG.start();
        SigmaNG.start();

        try {
            SigmaFS.join();
            SigmaNS.join();
            SigmaBlur.join();
            SigmaFG.join();
            SigmaNG.join();
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        outBmp.setPixels(SigmaNG.pixelsOut,0,picWidth,0,0,picWidth,picHeight);
        TimeInterval = System.currentTimeMillis() - javaStart;   //elapsed time calculation
        Log.d("TILTSHIFT_JAVA","Time Interval:"+TimeInterval);
        return outBmp;
    }
	
// Cpp implementation
    public static Bitmap tiltshift_cpp(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){

        long cppStart = System.currentTimeMillis();
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        //Call the function and pass variables needed for the C++ native implementation
        tiltshiftcppnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);
        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        TimeInterval = System.currentTimeMillis() - cppStart ;   //elapsed time calculation
        Log.d("TILTSHIFT_C++","Time Interval:"+TimeInterval);
        return outBmp;
    }

//Neon implementation
    public static Bitmap tiltshift_neon(Bitmap input, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3){
        long neonStart = System.currentTimeMillis();
        Bitmap outBmp = Bitmap.createBitmap(input.getWidth(), input.getHeight(), Bitmap.Config.ARGB_8888);
        int[] pixels = new int[input.getHeight()*input.getWidth()];
        int[] pixelsOut = new int[input.getHeight()*input.getWidth()];
        input.getPixels(pixels,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        tiltshiftneonnative(pixels,pixelsOut,input.getWidth(),input.getHeight(),sigma_far,sigma_near,a0,a1,a2,a3);
        outBmp.setPixels(pixelsOut,0,input.getWidth(),0,0,input.getWidth(),input.getHeight());
        TimeInterval = System.currentTimeMillis() - neonStart ;   //elapsed time calculation
        Log.d("TILTSHIFT_NEON","Time interval:"+TimeInterval);
        return outBmp;
    }

    public static int[] calK(float sigma) {
        // setting the radius to 2 Sigma
        int r = (int) Math.ceil(2 * sigma);
        int kernel_size = 2 * r + 1;
        int[] k = new int[kernel_size];             // initializing Kernel vector
        int m = -r;  
  
        for (int i = 0; i < kernel_size; i++) {
            k[i] = m++;
        }
        return k;
    }

    public static double[] GaussianWeights(int[] k, double sigma){
        int len = k.length;
        double[] weight = new double[len];
        for(int i=0; i<len; i++){
            //Calculate the weights in the gaussian distribution described in the pdf
            weight[i]=(Math.exp((-1*k[i]*k[i])/(2*sigma*sigma))/Math.sqrt(2*Math.PI*sigma*sigma));
        }
        return weight;
    }

    // Calculates the final weighted sum by gaussian weights
    // Compute the guassian blur for each color channel and add it to the pixels
    public static int Vector_1(String channel, int x, int y, int width, int[] k, double[] G, int[] pixel){
        double P=0;
        int len = k.length;
        for(int i=0; i<len; i++){
            if(y * width + x + k[i] < pixel.length) {
                //Computed the new value by calling the Vector_2 function
                P = P + G[i] * Vector_2(channel, x + k[i], y, width, k, G, pixel);
            }
        }
        return (int)P ;
    }

    public static double Vector_2(String channel, int x, int y, int width, int[] k, double[] G, int[] pixels){

        double q=0;
        int p;
        int len = k.length;
        //This switch-case uses the color scheme AA RR BB GG
        switch(channel){
            case "BB": for(int i=0; i<len; i++){					                //zero padding for edges
                if((y+k[i])<0 || x<0 || (y+k[i])*width+x >= pixels.length){
                    p=0;
                }
                else{
                    p=pixels[(y+k[i])*width+x];
                }
                int BB = p & 0xff;
                q+=BB*G[i];
            }
                return q;

            case "GG": for(int i=0; i<len; i++){
                if((y+k[i])<0 || x<0 || (y+k[i])*width+x>= pixels.length){             //checking the conditions for the edge cases
                
                    p=0;
                }
                else{           
                    p=pixels[(y+k[i])*width+x];
                }
                int GG = (p>>8) & 0xff;
                q+=GG*G[i];
            }
                return q;

            case "RR": for(int i=0; i<len; i++){
                if((y+k[i])<0 || x<0 || (y+k[i])*width+x>= pixels.length){
                    p=0;
                }
                else{
                    p=pixels[(y+k[i])*width+x];
                }
                int RR = (p>>16) &0xff;
                q+=RR*G[i];
            }
                return q;

            case "AA": for(int i=0; i<len; i++){
                if((y+k[i])<0 || x<0 || (y+k[i])*width+x>= pixels.length){			//checking the conditions for the edge cases
                    p=0;
                }
                else {
                    p=pixels[(y+k[i])*width+x];
                }
                int AA = (p>>24) &0xff;
                q+=AA*G[i];
            }
                return q;
        }
        return q;
    }
    /**
     * A native method that is implemented by the 'native-lib' native library,
     * which is packaged with this application.
     */
    public static native int tiltshiftcppnative(int[] inputPixels, int[] outputPixels, int width, int height, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3);
    public static native int tiltshiftneonnative(int[] inputPixels, int[] outputPixels, int width, int height, float sigma_far, float sigma_near, int a0, int a1, int a2, int a3);
}