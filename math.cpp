#include "math.h"

// add two vectors
Vector operator+(const Vector &v1, const Vector &v2) {
    return Vector(v1.x + v2.x, v1.y + v2.y, v1.z + v2.z);
}

// subtract two vectors
Vector operator-(const Vector &v1, const Vector &v2) {
    return Vector(v1.x - v2.x, v1.y - v2.y, v1.z - v2.z);
}

// element-wise multiply vectors
Vector operator*(const Vector &v1, const Vector &v2) {
    return Vector(v1.x * v2.x, v1.y * v2.y, v1.z * v2.z);
}

// element-wise divide vectors
Vector operator/(const Vector &v1, const Vector &v2) {
    if (v2.x == 0.0 || v2.y == 0.0 || v2.z == 0.0) {
        std::cerr << "Warning: Division by zero" << std::endl;
        return Vector(); 
    }
    return Vector(v1.x / v2.x, v1.y / v2.y, v1.z / v2.z);
}

// add a constant to a vector
Vector operator+(const Vector &v, double c) {
    return Vector(v.x + c, v.y + c, v.z + c);
}

// add a constant to a vector
Vector operator+(double c, const Vector &v) {
    return Vector(v.x + c, v.y + c, v.z + c);
}

// subtract a vector from a constant value
Vector operator-(double c, const Vector &v) {
    return Vector(c - v.x, c - v.y, c - v.z);
}

// subtract a constant from a vector
Vector operator-(const Vector &v, double c) {
    return Vector(v.x - c, v.y - c, v.z - c);
}

// scalar multiply a vector
Vector operator*(const Vector &v, double c) {
    return Vector(v.x * c, v.y * c, v.z * c);
}

// scalar multiply a vector
Vector operator*(double c, const Vector &v) {
    return Vector(v.x * c, v.y * c, v.z * c);
}

// scalar divide a vector
Vector operator/(const Vector &v, double c) {
    if (c == 0.0) {
        std::cerr << "Warning: Division by zero" << std::endl;
        return Vector(); 
    }
    return Vector(v.x / c, v.y / c, v.z / c);
}

// divide a scalar by vector values
Vector operator/(double c, const Vector &v) {
    if (v.x == 0.0 || v.y == 0.0 || v.z == 0.0) {
        std::cerr << "Warning: Division by zero" << std::endl;
        return Vector(); 
    }
    return Vector(c / v.x, c / v.y, c / v.z);
}

// Vector constructor
Vector::Vector(double xPos, double yPos, double zPos) : x{xPos}, y{yPos}, z{zPos} {}

// Vector constructor from a point
Vector::Vector(const Point &p): Vector(p.x, p.y, p.z) {}


// x getter                                                                     
double Vector::getX() const {
    return x;
}

// y getter                                                                     
double Vector::getY() const {
    return y;
}

// z getter                                                                     
double Vector::getZ() const {
    return z;
}

// print the vector components
void Vector::print() const {
    std::cout << "(" << x << ", " << y << ", " << z << ")" << std::endl;
}

// return the magnitude of the vector
double Vector::mag() const {
    return std::sqrt(std::pow(x, 2) + std::pow(y, 2) + std::pow(z, 2));
}

// normalize vector
void Vector::normalize() {
    double mag = this->mag();  

    if (mag == 0.0) {
        std::cerr << "Warning: Cannot normalize a zero vector" << std::endl;
        return; 
    }
    x /= mag;
    y /= mag;
    z /= mag;
}

// return a point representing the vector tip
Point Vector::getVectorAsPoint() const {
    return Point{x,y,z};
}
// return the dot product of two vectors
double dot(const Vector& v1, const Vector& v2) {
    return v1.getX() * v2.getX() + v1.getY() * v2.getY() + v1.getZ() * v2.getZ();
}

// clamp a value between two values
double clamp(double value, double min_value, double max_value) {
    return std::max(min_value, std::min(value, max_value));
}

// compute the normal of a point of a sphere
Vector computeSphereNormal(const Point &p1, const Point &c) {
    Vector normal(p1 - c);
    normal.normalize();
    return normal;
}

// Helper swap function
void swap(auto &a, auto &b) {
    auto temp = a;
    a = b;
    b = temp;
}

