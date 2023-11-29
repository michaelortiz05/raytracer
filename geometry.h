#ifndef GEOMETRY_H
#define GEOMETRY_H
#include <cuda_runtime.h>
#include "math.h"

// constants 
namespace constants {
    constexpr double alpha = 1.0;
};

// basic struct for point and point operations
struct Point {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;

    // Add two points
    Point operator+(const Point &p) const {
        return Point{x + p.x, y + p.y, z + p.z};
    }

    // Subtract two points
    Point operator-(const Point &p) const {
        return Point{x - p.x, y - p.y, z - p.z};
    }

    // Scalar multiplication
    Point operator*(double scalar) const {
        return Point{x * scalar, y * scalar, z * scalar};
    }

    // Scalar division
    Point operator/(double scalar) const {
        if (scalar == 0.0) {
            std::cerr << "Warning: Division by zero" << std::endl;
            return Point();
        }
        return Point{x / scalar, y / scalar, z / scalar};
    }
};

// // CUDA compatible point
// struct cudaPoint {
//     double x = 0.0;
//     double y = 0.0;
//     double z = 0.0;

//     __host__ __device__ cudaPoint operator+(const cudaPoint &p) const {
//         return Point{x + p.x, y + p.y, z + p.z};
//     }

//     __host__ __device__ cudaPoint operator-(const cudaPoint &p) const {
//         return Point{x - p.x, y - p.y, z - p.z};
//     }

//     __host__ __device__ cudaPoint operator*(double scalar) const {
//         return Point{x * scalar, y * scalar, z * scalar};
//     }

//     // Removed error handling for division by zero for CUDA compatibility
//     __host__ __device__ cudaPoint operator/(double scalar) const {
//         return Point{x / scalar, y / scalar, z / scalar};
//     }
// };

// vector class definining a vector and vector operations
class Vector {
    private:
        double x, y, z;
    public:
        Vector(double xPos = 0.0, double yPos = 0.0, double zPos = 0.0);
        Vector(const Point &p);
        double getX() const;
        double getY() const;
        double getZ() const;
        double mag() const;
        Point getVectorAsPoint() const;
        void print() const;
        void normalize();

        // Friend operator overloads
        friend Vector operator*(const Vector &v1, const Vector &v2);
        friend Vector operator/(const Vector &v1, const Vector &v2);
        friend Vector operator+(const Vector &v1, const Vector &v2);
        friend Vector operator+(const Vector &v, double c);
        friend Vector operator-(const Vector &v1, const Vector &v2);
        friend Vector operator+(double c, const Vector &v);
        friend Vector operator-(double c, const Vector &v);
        friend Vector operator-(const Vector &v, double c);
        friend Vector operator*(const Vector &v, double c);
        friend Vector operator*(double c, const Vector &v);
        friend Vector operator/(const Vector &v, double c);
        friend Vector operator/(double c, const Vector &v);
};

// cuda coordinates
struct __host__ __device__ cudaCoordinates{
    double x, y, z;
};

// cuda magnitude computation
double cudaMag(cudaCoordinates c);

// RGB color struct
struct __host__ __device__ Color {
    double r = 1.0;
    double g = 1.0;
    double b = 1.0;
    double alpha = 1.0;
};

// Sphere with center c, radius r, and a color
struct Sphere {
    Point c;
    double r = 0.0;
    Color color;
};

// Sun light source
struct Sun {
    Vector direction;
    Color c;
};

// cuda sun light source
struct __host__ __device__ cudaSun {
    cudaCoordinates direction;
    Color c;
};

#endif