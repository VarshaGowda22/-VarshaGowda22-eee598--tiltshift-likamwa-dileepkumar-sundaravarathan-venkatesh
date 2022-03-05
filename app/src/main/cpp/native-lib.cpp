#include <jni.h>
#include <string>
#include <cpu-features.h>
#include <thread>
#include <cmath>
#include <android/log.h>
#include <typeinfo>
#include <chrono>
using namespace std;
#include <arm_neon.h>
#include <iostream>
#include <cmath>
#include <android/log.h>
#include <jni.h>
#include <arm_neon.h>
#include <cmath>
int pixel_size;
jint* h;

//getSigma_neon function calculates sigma values for different values of y
float computeSigma_neon(int row, int a0, int a1, int a2, int a3, float32_t sigma_far, float32_t sigma_near){

    float multiplier = 0.0f;
    if (row<a0) { multiplier = sigma_far;}
    else if (row<a1) {multiplier = (sigma_far *(a1-row)+ 1.0f*(row-a0))/(a1-a0);}
    else if (row<=a2) {multiplier = 0.0f;}
    else if (row<a3) {multiplier = (1.0f*(a3-row)+ sigma_near*(row-a2))/(a3-a2);}
    else{ multiplier = sigma_near; }
    return multiplier;
}

//getK_neon function calculates kernel radius vector from sigma
int* CalK_neon(float sigma){
    int r=(int)ceil(3*sigma);
    int size=2*r+1;
    int* k=new int[size];             // k vector for G calculation
    int m=-r;
    for(int i=0; i<size ; i++){
        k[i]=m++;
    }
    return k;
}

//Calculates a R G B vector, placing each in a different lane from the pixel pointer and returning the results
uint32x4_t SeperateARGB(int* pixel){
    uint8_t* inputArrPtr = (uint8_t *)pixel;
    uint8x16x4_t pixelChannels = vld4q_u8(inputArrPtr);
    uint8x16_t A = pixelChannels.val[3];
    uint8x16_t R = pixelChannels.val[2];
    uint8x16_t G = pixelChannels.val[1];
    uint8x16_t B = pixelChannels.val[0];

    uint32x4_t ARGB = vdupq_n_u32(0);

    uint32_t AA = (uint32_t)vgetq_lane_u8(A,0);
    uint32_t RR = (uint32_t)vgetq_lane_u8(R,0);
    uint32_t GG = (uint32_t)vgetq_lane_u8(G,0);
    uint32_t BB = (uint32_t)vgetq_lane_u8(B,0);

    ARGB=vsetq_lane_u32(AA,ARGB,3);
    ARGB=vsetq_lane_u32(RR,ARGB,2);
    ARGB=vsetq_lane_u32(GG,ARGB,1);
    ARGB=vsetq_lane_u32(BB,ARGB,0);

    return ARGB;
}

// This function performs 1D-convolution on the shifted pixel arrays
float32x4_t Vector_2_neon(int x, int y, int width, int* k, int klen, double* G, int* pixels, int length){
    
    float32x4_t q=vdupq_n_f32(0);
    uint32x4_t p;

    for(int i=0; i<klen; i++){

        if((y+k[i])<0 || x<0 || (y+k[i])*width+x>= length) continue;//checking the conditions for the edge cases
        else {
            p= SeperateARGB(&pixels[(y+k[i])*width+x]);
        }
        //initialize a vector to hold the gaussian weights
        float32x4_t Gvector = vdupq_n_f32((float)G[i]);
        float32x4_t p_temp = vcvtq_f32_u32(p);
        q = vmlaq_f32(q, p_temp,Gvector);
    }
    return q;
}

//calculates the final weighted sum using the output of vector 2
uint32x4_t Vector_1_neon(int x, int y, int width, int* k, int klen, double* G, int* pixel, int length){
    float32x4_t P1 = vdupq_n_f32(0);
    for(int i=0; i<klen; i++){
        float32x4_t GVector = vdupq_n_f32((float)G[i]);
        if(y*width+x+k[i]<length)
            P1 = vmlaq_f32(P1, GVector, Vector_2_neon(x+k[i],y, width, k,klen, G, pixel, length));
    }
    uint32x4_t P = vcvtq_u32_f32(P1);
    return P;
}

// function to compute the gaussian weights using kernel variables
double* GaussianWeightCal(int* k, float sigma,int ksize){
    double* weight = new double[ksize];
    for(int i=0; i < ksize; i++){
        //Determine the weights as mentined in pdf of gaussian distribution
        weight[i]=( exp( (-1 * k[i] * k[i] ) / (2 * sigma * sigma) ) / sqrt( 2 * M_PI * sigma * sigma) );
    }
    return weight;
}


float Vector_2(string channel, int x, int y, int width, int* k, double* G, int* pixels,int ksize) {
    double q = 0;
    int p;
    //This switch-case uses the color scheme AA RR BB GG
    if (channel == "BB") {
        for (int i = 0; i < ksize; i++) {
            //zero padding for edges
            if ((y + k[i]) < 0 || x < 0 || (y + k[i]) * width + x >= pixel_size) { p = 0; }				//checking the conditions for the edge cases
            else { p = pixels[(y + k[i]) * width + x]; }
            int BB = p & 0xff;
            q += BB * G[i];
        }
    }
    else if (channel == "GG") {
        for (int i = 0; i < ksize; i++) {
            if ((y + k[i]) < 0 || x < 0 || (y + k[i]) * width + x >= pixel_size) { p = 0; }
            else { p = pixels[(y + k[i]) * width + x]; }
            int GG = (p >> 8) & 0xff;				  													 // make a 8 bit right shift
            q += GG * G[i];
        }
    }
    else if (channel == "RR"){
        for (int i = 0; i < ksize; i++) {
            if ((y + k[i]) < 0 || x < 0 || (y + k[i]) * width + x >= pixel_size) { p = 0; }
            else { p = pixels[(y + k[i]) * width + x]; }
            int RR = (p >> 16) & 0xff;										 							// make a 8 bit right shift
            q += RR * G[i];
        }
    }
    else if (channel == "AA"){
        for(int i=0; i< ksize; i++){
            if((y + k[i]) < 0 || x<0 || (y + k[i]) * width + x >= pixel_size){p = 0;}
            else {p = pixels[( y + k[i] ) * width + x]; }
            int AA = (p>>24) &0xff;
            q += AA * G[i];
        }
    }
    else{
        __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_NO CHANNEL", "Good Day");
    }
    return q;
}

//calculates the final weighted sum using the output of vector 
int Vector_1(string channel, int x, int y, int width, int* k, double* G, int* pixel,int ksize){
    double P=0;
    for(int i=0; i < ksize; i++){
        if( y * width + x + k[i] < pixel_size) {
            P = P + G[i] * Vector_2(channel, x + k[i], y, width, k, G, pixel, ksize);
        }
    }
    return (int)P ;
}

// Thread Manipulation function
void Sigma_FN(string threadName,float sigma,int a0,int a1,int a2,int a3,int* pixels,int height,int width) {
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_THREAD",
            "creating Thread:%s,Sigma:%f,a0:%d,a1:%d,a2:%d,a3:%d",&threadName,sigma,a0,a1,a2,a3);

    //Region 1 - between row 0 and a0
    //
    //In the sigma_far region between the starting index and a0, the sigma value remains unchanged
    if (threadName == "Thread-sigma-far-solid") {
        if (sigma >= 0.6){
            // Determine the Kernel radius and assign indices using the sigma value
            int r = (int) ceil(2 * sigma);
            int kernel_size = 2 * r + 1;
            int* k = new int[kernel_size];             				// initializing Kernel vector
            int m = -r; 
            for (int i = 0; i < kernel_size; i++) {					// Initiate the kernel with increasing decreasing indexes
                k[i] = m++;
            }
            //Determine the gaussian weights using the kernel indices and the sigma value
            double* kMatrix = GaussianWeightCal(k, sigma,kernel_size);
            for (int i=0; i < a0; i++){
                //Compute the result of convolution with Gaussian kernel for each R,G,B,A color channel
                for(int j=0; j < width; j++){
                    int pBB = Vector_1("BB", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pRR = Vector_1("RR", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pGG = Vector_1("GG", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pAA = Vector_1("AA", j, i, width, k, kMatrix, pixels,kernel_size);
                    h[i * width + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                }
            }
        }
            // If sigma is below the  threshold value the pixels are left untouched
        else {
            for (int i = 0; i < a0; i++) {
                for (int j = 0; j < width; j++) {
                    h[i * width + j] = pixels[i * width + j];
                }
            }
        }
    }
        //Region 2 - between row a0 and a1

    else if(threadName == "Thread-sigma-far-gradual"){
        float sigmaY;
        for (int i = a0 ; i < a1 ; i++) {
            sigmaY = sigma * ((float) (a1-i)/(a1-a0));
            if(sigmaY >= 0.6) {
                int r = (int) ceil(2 * sigmaY);
                int kernel_size = 2 * r + 1;
                int* k = new int[kernel_size];             // initializing Kernel vector
                int m = -r;  
                for (int i = 0; i < kernel_size; i++) {
                    k[i] = m++;
                }
                //Compute the gaussian weights using the kernel indices and the sigma value
                double* kMatrix = GaussianWeightCal(k, sigmaY,kernel_size);  								//Calculate the Gaussian Matrix
                for (int j = 0; j < width; j++) {
                    int pBB = Vector_1("BB", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pRR = Vector_1("RR", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pGG = Vector_1("GG", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pAA = Vector_1("AA", j, i, width, k, kMatrix, pixels,kernel_size);
                    h[i * width + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                }
            }
            else{
                for(int z = i ; z <= a1 ; z++){
                    // Copy the new pixel values to pixelsOut
                    for (int j = 0; j < width; j++) {
                        h[z * width + j] = pixels[z * width + j];
                    }
                }
                break;
            }
        }
    }
        //
        //Region 3 - between row a1 and a2
        //
        //In this region the image pixels are left untouched
    else if (threadName == "Thread-sigma-no-blur") {
        for (int j = a1; j <= a2; j++) {
            for (int i = 0; i < width; i++) {
                h[j * width + i] = pixels[j * width + i];
            }
        }
    }
        //
        //Region 4 - between row a2 and a3
        //
        //In this region the sigma value gradually increase till it rises above the threshold value
    else if(threadName == "Thread-sigma-near-gradual") {
        float sigmaX;
        for (int i = a3; i > a2; i--) {
            sigmaX = sigma * ((float)(i-a2)/(a3-a2));
            if (sigmaX >= 0.6) {
                int r = (int) ceil(2 * sigmaX);
                int kernel_size = 2 * r + 1;
                int* k = new int[kernel_size];            					 // initializing Kernel vector
                int m = -r; 
                for (int i = 0; i < kernel_size; i++) {
                    k[i] = m++;
                }
                double* kMatrix = GaussianWeightCal(k, sigmaX,kernel_size);  //Calculate the Gaussian Matrix
                for (int j = 0; j < width; j++) {
                    //Compute the result of convolution with Gaussian kernel for each R,G,B,A color channel
                    int pBB = Vector_1("BB", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pRR = Vector_1("RR", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pGG = Vector_1("GG", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pAA = Vector_1("AA", j, i, width, k, kMatrix, pixels,kernel_size);
                    h[i * width + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 | (pBB & 0xff);
                }
            }
            else {
                for(int z = i; z > a2 ; z--) {
                    for (int j = 0; j < width; j++) {
                        h[z * width + j] = pixels[z * width + j];
                    }
                }
                break;
            }
        }
    }
        
        //Region 5 - between row a3 and final

    else if(threadName == "Thread-sigma-near-solid") {
        if (sigma >= 0.6) {
            int r = (int) ceil(2 * sigma);
            int kernel_size = 2 * r + 1;
            int* k = new int[kernel_size];             // initializing Kernel vector
            int m = -r;  
            for (int i = 0; i < kernel_size; i++) {
                k[i] = m++;
            }
            //Determine the gaussian weights using the kernel indices and the sigma value
            double *kMatrix = GaussianWeightCal(k, sigma,kernel_size);
            for (int i = a3; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    int pBB = Vector_1("BB", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pRR = Vector_1("RR", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pGG = Vector_1("GG", j, i, width, k, kMatrix, pixels,kernel_size);
                    int pAA = Vector_1("AA", j, i, width, k, kMatrix, pixels,kernel_size);
                    h[i * width + j] = (pAA & 0xff) << 24 | (pRR & 0xff) << 16 | (pGG & 0xff) << 8 |
                                       (pBB & 0xff);
                }
            }
        }
        else {
            for (int i = a3; i < height; i++) {
                for (int j = 0; j < width; j++) {
                    h[i * width + j] = pixels[i * width + j];
                }
            }
        }
    }
}

extern "C"
JNIEXPORT jint JNICALL
Java_edu_asu_ame_meteor_speedytiltshift2022_SpeedyTiltShift_tiltshiftcppnative(JNIEnv *env,
                                                                               jobject instance,
                                                                               jintArray inputPixels_,
                                                                               jintArray outputPixels_,
                                                                               jint width,
                                                                               jint height,
                                                                               jfloat sigma_far,
                                                                               jfloat sigma_near,
                                                                               jint a0, jint a1,
                                                                               jint a2, jint a3) {
//    Get the input and output pixels from the java environment
    jint *pixels = env->GetIntArrayElements(inputPixels_, NULL);
    jint *outputPixels = env->GetIntArrayElements(outputPixels_, NULL);
    h = new jint[height*width];
    pixel_size = height*width;

    int scale = 1;
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_MAIN","height:%d, width:%d ,pixel_size:%d",height,width,height*width);
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_MAIN",
            "sigma far:%f,sigma near:%f,a0:%d,a1:%d,a2:%d,a3:%d",sigma_far*scale,sigma_near*scale,a0,a1,a2,a3);
 
    try {
        std::thread SigmaFarSolid(Sigma_FN, "Thread-sigma-far-solid", sigma_far * scale, a0,a1, a2, a3, pixels, height, width);
        std::thread SigmaFarGradual(Sigma_FN, "Thread-sigma-far-gradual", sigma_far * scale,a0, a1, a2, a3, pixels, height, width);
        std::thread SigmaNoBlur(Sigma_FN, "Thread-sigma-no-blur", sigma_far, a0, a1, a2, a3,pixels, height, width);
        std::thread SigmaNearGradual(Sigma_FN, "Thread-sigma-near-gradual",sigma_near * scale, a0, a1, a2, a3, pixels, height, width);
        std::thread SigmaNearSolid(Sigma_FN, "Thread-sigma-near-solid", sigma_near * scale,a0, a1, a2, a3, pixels, height, width);

        SigmaFarSolid.join();
        SigmaFarGradual.join();
        SigmaNoBlur.join();
        SigmaNearGradual.join();
        SigmaNearSolid.join();
    }
    catch (std::exception& e)
    {
        __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_C++_THREAD TRY","%s",e.what());
    }

    env->ReleaseIntArrayElements(inputPixels_, pixels, 0);
    env->ReleaseIntArrayElements(outputPixels_, h, 0);
    return 0;
}

extern "C"
JNIEXPORT jint JNICALL
Java_edu_asu_ame_meteor_speedytiltshift2022_SpeedyTiltShift_tiltshiftneonnative(JNIEnv *env,
                                                                                jclass instance,
                                                                                jintArray inputPixels_,
                                                                                jintArray outputPixels_,
                                                                                jint width,
                                                                                jint height,
                                                                                jfloat sigma_far,
                                                                                jfloat sigma_near,
                                                                                jint a0, jint a1,
                                                                                jint a2, jint a3) {

    jint *pixels = env->GetIntArrayElements(inputPixels_, NULL);
    jint *outputPixels = env->GetIntArrayElements(outputPixels_, NULL);
    long pixel_length = env->GetArrayLength(inputPixels_);
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_NEON_MAIN","height:%d, width:%d ,pixel_size:%d",height,width,height*width);
    __android_log_print(ANDROID_LOG_INFO, "TILTSHIFT_NEON_MAIN", "length: %d %d %d %d %d",pixel_length,a0,a1,a2,a3);
    int flag = 0, count = 0,a4 = 0;
    for (int y= 0; y < height; y++){
        float sigma =0;
        sigma=computeSigma_neon(y,a0,a1,a2,a3,sigma_far,sigma_near);
        if(sigma<=0.6){
            if (flag == 0) {
                a4 = y;
                flag = 1;
            }
            count++;
            continue;
        }
        int r = (int)ceil(2.5*sigma);
        int kernel_size = 2*r+1;
        int* k= new int[kernel_size];
        //calculate the Kernel radius and assign indices using the sigma value
        k= CalK_neon(sigma);
        //calculate the gaussian weights using the kernel indices and the sigma value
        double* kMatrix = new double[kernel_size];
        kMatrix = GaussianWeightCal(k, sigma, kernel_size);
        for (int x = 0; x < width; x++) {
            // Convolution for all the channels with the gaussian weight matrix
            uint32x4_t ARGB = Vector_1_neon(x, y, width, k, kernel_size, kMatrix, pixels, pixel_length);
            int partA = vgetq_lane_u32(ARGB, 3);
            int partR = vgetq_lane_u32(ARGB, 2);
            int partG = vgetq_lane_u32(ARGB, 1);
            int partB = vgetq_lane_u32(ARGB, 0);
            outputPixels[y * width + x] = (partA & 0xff) << 24 | (partR & 0xff) << 16 | (partG & 0xff) << 8 | (partB & 0xff);
        }
        delete[] k;
        delete[] kMatrix;
    }
    for (int j = a4;j <= a4+count; j++) {
        for (int i=0; i<width ;i++) {
            outputPixels[j * width + i] = pixels[j * width + i];
        }
    }
    env->ReleaseIntArrayElements(inputPixels_, pixels, 0);
    env->ReleaseIntArrayElements(outputPixels_, outputPixels, 0);
    return 0;
}