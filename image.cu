#include "image.h"

// printing overload for color
std::ostream &operator<<(std::ostream &out, const Color &c)
{
    out << "(" << c.r << ", " << c.g << ", " << c.b << ')';
    return out;
}

// Image constructor
Image::Image(int w, int h, std::string n) : width{w}, height{h}, name{n}
{
    png.resize(height * width * 4);
    maxDim = std::max(width, height);
    currentSun = nullptr;
    eye = new Point;
    forward = new Vector(0, 0, -1);
    right = new Vector(1, 0, 0);
    up = new Vector(0, 1, 0);
}

// Image deconstructor
Image::~Image()
{
    delete currentSun;
    delete eye;
    delete forward;
    delete right;
    delete up;
}

// Height getter
int Image::getHeight() { return height; }

// Width getter
int Image::getWidth() { return width; }

// Coor getter
Color Image::getColor() { return currentColor; }

// Name getter
std::string const &Image::getName() { return name; }

// Png getter
std::vector<unsigned char> const &Image::getPng() { return png; }

// Set the color
void Image::setColor(double r, double g, double b, double a)
{
    currentColor.r = clamp(r, 0.0, 1.0);
    currentColor.g = clamp(g, 0.0, 1.0);
    currentColor.b = clamp(b, 0.0, 1.0);
    currentColor.alpha = clamp(a, 0.0, 1.0);
}

// add sphere to list of objects
void Image::addObject(double x, double y, double z, double r)
{
    objects.push_back(Sphere{Point{x, y, z}, r, this->currentColor});
}

// Set the current sun
void Image::setSun(double x, double y, double z)
{
    if (currentSun == nullptr)
    {
        currentSun = new Sun;
        currentSun->c = currentColor;
    }
    Vector d(x, y, z);
    d.normalize();
    currentSun->direction = d;
    currentSun->c = currentColor;
}

// Print the current sun
void Image::printSun()
{
    if (currentSun != nullptr)
    {
        std::cout << "Color: " << currentSun->c << std::endl;
        std::cout << "Direction: " << std::endl;
        currentSun->direction.print();
    }
}

// Cast rays and draw the scene
void Image::castRays()
{
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double sx = (2.0 * x - width) / maxDim;
            double sy = (height - 2.0 * y) / maxDim;
            Vector direction = *forward + sx * *right + sy * *up;
            direction.normalize();
            const Ray ray = Ray{*eye, direction};
            Intersection intersection = getSphereCollision(ray);
            if (intersection.found == true && intersection.t > 0.0)
            {
                Vector normal = computeSphereNormal(intersection.p, intersection.center);
                computeColor(normal, intersection.c, intersection.p);
                colorPixel(x, y, intersection.c);
            }
        }
    }
}

void Image::castRays_parallel()
{
#pragma omp parallel for
    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double sx = (2.0 * x - width) / maxDim;
            double sy = (height - 2.0 * y) / maxDim;
            Vector direction = *forward + sx * *right + sy * *up;
            direction.normalize();
            const Ray ray = Ray{*eye, direction};
            Intersection intersection = getSphereCollision(ray);
            if (intersection.found == true && intersection.t > 0.0)
            {
                Vector normal = computeSphereNormal(intersection.p, intersection.center);
                computeColor(normal, intersection.c, intersection.p);
                colorPixel(x, y, intersection.c);
            }
        }
    }
}

__global__ void castRaysKernel(Vector *forward, Vector *right, Vector *up, Point *eye, int width, int height, double maxDim, Color *output, Image *image)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    double sx = (2.0 * x - width) / maxDim;
    double sy = (height - 2.0 * y) / maxDim;
    Vector direction = *forward + sx * *right + sy * *up;
    direction.normalize();
    Ray ray = Ray{*eye, direction};
    Intersection intersection = image->getSphereCollision(ray);
    
    if (intersection.found == true && intersection.t > 0.0)
    {
        Vector normal = computeSphereNormal(intersection.p, intersection.center);
        image->computeColor(normal, intersection.c, intersection.p);
        output[y * width + x] = intersection.c; // Storing result in the output array
    }
}


void Image::castRays_CUDA()
{
    Color *device_output;
    Vector *device_forward, *device_right, *device_up;
    Point *device_eye;
    Image *device_image;

    // Allocate memory on the device
    cudaMalloc(&device_output, width * height * sizeof(Color));
    cudaMalloc(&device_forward, sizeof(Vector));
    cudaMalloc(&device_right, sizeof(Vector));
    cudaMalloc(&device_up, sizeof(Vector));
    cudaMalloc(&device_eye, sizeof(Point));
    cudaMalloc(&device_image, sizeof(Image));

    // Copy data from host to device
    cudaMemcpy(device_forward, forward, sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(device_right, right, sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(device_up, up, sizeof(Vector), cudaMemcpyHostToDevice);
    cudaMemcpy(device_eye, eye, sizeof(Point), cudaMemcpyHostToDevice);
    cudaMemcpy(device_image, this, sizeof(Image), cudaMemcpyHostToDevice);

    // Define block size and grid size
    dim3 blockSize(16, 16); // Adjust these values as needed for your GPU
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Launch the kernel
    castRaysKernel<<<gridSize, blockSize>>>(device_forward, device_right, device_up, device_eye, width, height, maxDim, device_output, device_image);

    // Copy the result back to host
    Color *host_output = new Color[width * height];
    cudaMemcpy(host_output, device_output, width * height * sizeof(Color), cudaMemcpyDeviceToHost);

    // Use host_output for further processing or displaying the image

    // Free device memory
    cudaFree(device_output);
    cudaFree(device_forward);
    cudaFree(device_right);
    cudaFree(device_up);
    cudaFree(device_eye);

    // Free host memory
    delete[] host_output;
}

// return the ray-sphere collision
__host__ __device__ Intersection Image::getSphereCollision(const Ray &ray) const
{
    Intersection intersection;
    for (auto &object : objects)
    {
        Vector diff(object.c - ray.origin);
        bool inside = std::pow(diff.mag(), 2.0) < std::pow(object.r, 2.0);
        double tc = dot(diff, ray.direction) / ray.direction.mag();
        if (!inside && tc < 0)
            continue;
        Point d = ray.origin + tc * ray.direction.getVectorAsPoint() - object.c;
        double d2 = std::pow(Vector(d).mag(), 2.0);
        if (!inside && std::pow(object.r, 2.0) < d2)
            continue;
        double tOffset = std::sqrt(std::pow(object.r, 2) - d2) / ray.direction.mag();
        double t = 0.0;
        intersection.found = true;
        t = inside ? tc + tOffset : tc - tOffset;
        if (t < intersection.t)
        {
            intersection.t = t;
            intersection.c = object.color;
            intersection.center = object.c;
        }
    }
    if (intersection.found == true)
        intersection.p = intersection.t * ray.direction.getVectorAsPoint() + ray.origin;
    return intersection;
}

// helper function to convert color space
void Image::convertLinearTosRGB(Color &c)
{
    const double factor = 0.0031308;
    c.r = clamp(c.r, 0.0, 1.0);
    c.g = clamp(c.g, 0.0, 1.0);
    c.b = clamp(c.b, 0.0, 1.0);
    c.r = c.r <= factor ? 12.92 * c.r : 1.055 * std::pow(c.r, (1.0 / 2.4)) - 0.055;
    c.g = c.g <= factor ? 12.92 * c.g : 1.055 * std::pow(c.g, (1.0 / 2.4)) - 0.055;
    c.b = c.b <= factor ? 12.92 * c.b : 1.055 * std::pow(c.b, (1.0 / 2.4)) - 0.055;
}

// compute color with lambert shading
__host__ __device__ void Image::computeColor(Vector &normal, Color &c, Point &p)
{
    if (currentSun != nullptr && !isInShadow(p))
    {
        Vector eyeDir(p - *eye);
        eyeDir.normalize();
        if (dot(eyeDir, normal) > 0.0)
            normal = normal * -1;
        currentSun->direction.normalize();
        double lambert = std::max(dot(normal, currentSun->direction), 0.0);
        c.r *= lambert * currentSun->c.r;
        c.g *= lambert * currentSun->c.g;
        c.b *= lambert * currentSun->c.b;
    }
    else
    {
        c.r = 0.0;
        c.g = 0.0;
        c.b = 0.0;
    }
}

// print the set of objects in the scene
void Image::printObjects()
{
    for (auto &object : objects)
    {
        std::cout << "Point: (";
        std::cout << object.c.x << ", ";
        std::cout << object.c.y << ", ";
        std::cout << object.c.z << ")" << std::endl;
        std::cout << "Radius: " << object.r << std::endl;
        std::cout << "Color: (" << object.color.r << ", ";
        std::cout << object.color.g << ", ";
        std::cout << object.color.b << ")" << std::endl;
    }
}

// check if a point is in shadow
__host__ __device__ bool Image::isInShadow(const Point &intersection)
{
    constexpr double bias = 1e-6;
    Vector biasVector = currentSun->direction * bias;
    Ray shadowRay{intersection + biasVector.getVectorAsPoint(), currentSun->direction};
    Intersection i = getSphereCollision(shadowRay);
    return i.found;
}

// color pixel at location
void Image::colorPixel(int x, int y, Color &c)
{
    convertLinearTosRGB(c);
    png[((y * width) + x) * 4 + 0] = static_cast<unsigned char>(c.r * 255.0);
    png[((y * width) + x) * 4 + 1] = static_cast<unsigned char>(c.g * 255.0);
    png[((y * width) + x) * 4 + 2] = static_cast<unsigned char>(c.b * 255.0);
    png[((y * width) + x) * 4 + 3] = static_cast<unsigned char>(c.alpha * 255.0);
}

