#include "testFunc.hpp"

int main() {
    
    uint width,
         height,
         width2,
         height2;
         
    unsigned char *img1 = loadPPM("../test/images/4emp.ppm", &width, &height),          // 1337x965
                  *img2 = loadPPM("../test/images/immagine.ppm", &width2, &height2);    // 640x360  
    
    test(img1, img2, &width, &height, &width2, &height2);

    return 0;
}
