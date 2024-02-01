#include "testFunc.hpp"

int main()
{

    uint width,
            height,
            width2,
            height2;

    char format1[3];
    char format2[3];

    unsigned char *img1 = loadPPM("../test/images/4emp.ppm", &width, &height, format1),          // 1337x965
    *img2 = loadPPM("../test/images/immagine.ppm", &width2, &height2, format2);    // 640x360

    test(img1, img2, &width, &height, &width2, &height2);

    return 0;
}
