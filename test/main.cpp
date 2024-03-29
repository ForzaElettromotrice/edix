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

    uint oWidth;
    uint oHeight;
    unsigned char *oImg = colorFilterSerial(img1, width1, height1, channels1, 255, 0, 0, 0, &oWidth, &oHeight);
    writeImage("images/out.ppm", oImg, oWidth, oHeight, channels1);

    testAccuracy(img1, img2, img3, width1, height1, width2, height2, width3, height3);
    testPerformance(img1, img2, width1, height1, channels1, width2, height2, channels2);


    free(img1);
    free(img2);
    free(img3);

    return 0;
}
