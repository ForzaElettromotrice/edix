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
    unsigned char *img1 = loadImage((char *) "images/4emp.ppm", &width1, &height1, &channels1);          // 1337x965
    unsigned char *img2 = loadImage((char *) "images/immagine.ppm", &width2, &height2, &channels2);    // 640x360
    unsigned char *img3 = loadImage((char *) "images/gray.ppm", &width3, &height3, &channels3);          // 1337x965


    testAccuracy(img1, img2, img3, width1, height1, width2, height2, width3, height3);

//    uint oWidth;
//    uint oHeight;
//
//    auto start = std::chrono::high_resolution_clock::now();
//    unsigned char *oImg = grayscaleSerial(img1, width1, height1, &oWidth, &oHeight);
//    auto end = std::chrono::high_resolution_clock::now();
//    writeImage((char *) "images/gray.ppm", oImg, oWidth, oHeight, 1);
//
//    uint delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
//    printf("Time: %u\n", delta);
//
//    free(oImg);
//    free(img1);
//    free(img2);

    return 0;
}
