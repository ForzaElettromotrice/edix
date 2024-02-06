#include "testFunc.hpp"

int main()
{

    uint width1;
    uint height1;
    uint channels1;
    uint width2;
    uint height2;
    uint channels2;
    unsigned char *img1 = loadImage((char *) "images/out.ppm", &width1, &height1, &channels1);          // 1337x965
    unsigned char *img2 = loadImage((char *) "images/immagine.ppm", &width2, &height2, &channels2);    // 640x360

//    test(img1, img2, &width1, &height1, &width2, &height2, &channels1, &channels2);

    uint oWidth;
    uint oHeight;

    auto start = std::chrono::high_resolution_clock::now();
    unsigned char *oImg = scaleCudaBilinear(img1, width1, height1, channels1, 4, true, &oWidth, &oHeight, false);
    auto end = std::chrono::high_resolution_clock::now();
    writeImage((char *) "images/out2.ppm", oImg, oWidth, oHeight, channels1);

    uint delta = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    printf("Time: %u\n", delta);

    free(oImg);
    free(img1);
    free(img2);

    return 0;
}
