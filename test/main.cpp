#include "main.hpp"

int main()
{

    uint width1;
    uint height1;
    uint channels1;
    uint width2;
    uint height2;
    uint channels2;
    uint width3;
    uint height3;
    uint channels3;
    unsigned char *img1 = loadImage((char *) "images/sfondo.ppm", &width1, &height1, &channels1);          // 1920x1080
    unsigned char *img2 = loadImage((char *) "images/immagine.ppm", &width2, &height2, &channels2);    // 640x360
    unsigned char *img3 = loadImage((char *) "images/gray.ppm", &width3, &height3, &channels3);          // 1337x965


//    testAccuracy(img1, img2, img3, width1, height1, width2, height2, width3, height3);
    testPerformance(img1, img2, width1, height1, channels1, width2, height2, channels2);
//    uint oWidth;
//    uint oHeight;
//
//    auto start = std::chrono::high_resolution_clock::now();
//    unsigned char *oImg = colorFilterCuda(img2, width2, height2, channels2, 50, 200, 200, 150, &oWidth, &oHeight);
//
//
//    auto end = std::chrono::high_resolution_clock::now();
//    writeImage((char *) "images/accuracy.ppm", oImg, oWidth, oHeight, channels1);
//
//    uint delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//    printf("Time: %u\n", delta);
//
//    free(oImg);
    free(img1);
    free(img2);
    free(img3);

    return 0;
}
