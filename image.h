#ifndef IMAGE_H
#define IMAGE_H
#include "math.h"
#include <cuda_runtime.h>
std::ostream& operator<<(std::ostream& out, const Color& c);

// Image class for rendering the scene
class Image {
    private:
        int width;
        int height;
        double maxDim;
        std::string name;
        Color currentColor;
        std::vector<unsigned char> png;
        std::vector<Sphere> objects;
        Sun* currentSun;
        Point* eye;
        Vector* forward;
        Vector* right;
        Vector* up;
        void colorPixel(int x, int y, Color &c);
    public:
        Image(int w = 0, int h = 0, std::string n = ""); 
        ~Image();
        __host__ __device__ Intersection getSphereCollision(const Ray &ray) const;
        __host__ __device__ void computeColor(Vector& normal, Color& c, Point& p);
        __host__ __device__ bool isInShadow(const Point &intersection);
        int getHeight();
        Color getColor();
        int getWidth();
        std::vector<unsigned char> const &getPng();
        std::string const &getName();
        void setColor(double r, double g, double b, double a = constants::alpha);
        void addObject(double x, double y, double z, double r);
        void setSun(double x, double y, double z);
        void printSun();
        void castRays();
        void castRays_parallel();
        void printObjects();
        static void convertLinearTosRGB(Color &c);
        void castRays_CUDA();
};
#endif
