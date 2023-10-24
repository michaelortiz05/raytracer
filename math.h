#ifndef MATH_H
#define MATH_H
#include <iostream>
#include <vector>
#include <algorithm>
#include <iterator>
#include <cmath>
#include <limits>
#include "geometry.h"

struct Ray {
    Point origin;
    Vector direction;
};

struct Intersection {
    Point p;
    double t = std::numeric_limits<double>::max();
    double found = false;
    Color c;
    Point center;
};

inline Point operator*(double scalar, const Point &p) {
    return Point{p.x * scalar, p.y * scalar, p.z * scalar};
}

double dot(const Vector& v1, const Vector& v2);
Vector computeSphereNormal(const Point &p1, const Point &c);
double clamp(double value, double min_value, double max_value);
void swap(auto &a, auto &b);
#endif