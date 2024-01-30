#include "testFunc.hpp"

int main() {
    
    uint width,
         height,
         width2,
         height2;
         
    unsigned char *img1 = loadPPM("./images/4emp.ppm", &width, &height), 
                  *img2 = loadPPM("./images/immagine.ppm", &width2, &height2);
    
    test(img1, img2, &width, &height, &width2, &height2);

    return 0;
}
