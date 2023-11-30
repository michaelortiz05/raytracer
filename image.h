#ifndef IMAGE_H
#define IMAGE_H
#include "math.h"
#include <cfloat>
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
        Sun* currentSun;
        Point* eye;
        Vector* forward;
        Vector* right;
        Vector* up;
        Intersection getSphereCollision(const Ray &ray) const;
        void colorPixel(int x, int y, Color &c);
        void computeColor(Vector& normal, Color& c, Point& p);
        bool isInShadow(const Point &intersection);
    public:
        Image(int w = 0, int h = 0, std::string n = ""); 
        ~Image();
        std::vector<Sphere> objects;
        int getHeight();
        Point* getEye();
        Vector* getForward();
        Vector* getRight();
        Vector* getUp();
        Color getColor();
        int getWidth();
        double getMaxDim();
        Sun* getSun();
        std::vector<unsigned char> const &getPng();
        std::string const &getName();
        void setColor(double r, double g, double b, double a = constants::alpha);
        void addObject(double x, double y, double z, double r);
        void setSun(double x, double y, double z);
        void printSun();
        void castRays();
        void printObjects();
        static void convertLinearTosRGB(Color &c);
};

// cuda compatible image struct
struct cudaImage {
    int width;
    int height;
    double maxDim;
    unsigned char* png;
    Color currentColor;
    cudaSun currentSun;
    cudaCoordinates eye;
    cudaCoordinates forward;
    cudaCoordinates right;
    cudaCoordinates up;
    cudaSphere* spheres;
    int numSpheres;
};

void __device__ convertLinearTosRGB(Color &c);

// convert image to cuda compatible image
void convertImageToCudaImage(Image &i, cudaImage &ci); // host

// convert sun to cuda sun
void convertSunToCudaSun(Sun &s, cudaSun &cs);

// cuda raytracer setup
void cudaRaytracer(cudaImage *ci);

// cuda intersection
struct __device__ cudaIntersection {
    cudaCoordinates p;
    double t = DBL_MAX;
    bool found = false;
    Color c;
    cudaCoordinates center;
};

// cuda get collisions
cudaIntersection __device__ getSphereCollision(cudaCoordinates &origin, cudaCoordinates &direction);

// cuda kernel
__global__ void castRaysKernel(cudaImage* image);

#endif
