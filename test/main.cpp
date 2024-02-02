#include "testFunc.hpp"

int main()
{

    uint width,
            height,
            width2,
            height2,
            channels1,
            channels2;
    unsigned char *img1 = loadImage((char *) "../test/images/4emp.ppm", &width, &height, &channels1),          // 1337x965
    *img2 = loadImage((char *) "../test/images/immagine.ppm", &width2, &height2, &channels2);    // 640x360

    test(img1, img2, &width, &height, &width2, &height2, &channels1, &channels2);

    return 0;
}
