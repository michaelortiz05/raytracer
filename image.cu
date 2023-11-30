#include "image.h"
#include "math.h"
#include <cmath>
/*******CUDA*********/
__constant__ cudaSun constSun;
__constant__ int height;
__constant__ int width;
__constant__ cudaCoordinates eye;
__constant__ cudaCoordinates forward;
__constant__ cudaCoordinates right;
__constant__ cudaCoordinates up;
__constant__ int maxDim;
__constant__ int numSpheres;

// compute mag of cuda coordinates
double __host__  __device__ cudaMag(cudaCoordinates &c) {
    return sqrt(c.x * c.x + c.y * c.y + c.z * c.z);
}
// cuda clamp
double __device__ cudaClamp(double value, double min_value, double max_value) {
    return max(min_value, min(value, max_value));
}
// cuda SRGB conversion
void __device__ cudaConvertLinearTosRGB(Color &c)
{
    const double factor = 0.0031308;
    c.r = cudaClamp(c.r, 0.0, 1.0);
    c.g = cudaClamp(c.g, 0.0, 1.0);
    c.b = cudaClamp(c.b, 0.0, 1.0);
    c.r = c.r <= factor ? 12.92 * c.r : 1.055 * std::pow(c.r, (1.0/2.4)) - 0.055;
    c.g = c.g <= factor ? 12.92 * c.g : 1.055 * std::pow(c.g, (1.0/2.4)) - 0.055;
    c.b = c.b <= factor ? 12.92 * c.b: 1.055 * std::pow(c.b, (1.0/2.4)) - 0.055;
}

// normalize cudaCoordinates
void __host__  __device__ cudaNormalize(cudaCoordinates &c) {
    double mag = cudaMag(c);
    c.x /= mag;
    c.y /= mag;
    c.z /= mag;
}

cudaCoordinates __device__ cudaNormalizev(cudaCoordinates &c) {
    double mag = cudaMag(c);
    cudaCoordinates cn = {c.x /= mag, c.y /= mag, c.z /= mag};
    return cn;
}

// cuda dot
double __device__ cudaDot(cudaCoordinates &c1, cudaCoordinates &c2) {
    return c1.x * c2.x + c1.y * c2.y + c1.z * c2.z;
}

// cuda intersection
cudaIntersection __device__ cudaGetSphereCollision(cudaSphere* spheres, int numSpheres, cudaCoordinates &origin, cudaCoordinates &direction) {
    cudaIntersection intersection;
    for (int i = 0; i < numSpheres; i++) {
        cudaSphere s = spheres[i];
        cudaCoordinates diff = s.c - origin;
        bool inside = pow(cudaMag(diff), 2.0) < pow(s.r, 2.0);
        double tc = cudaDot(diff, direction) / cudaMag(direction);
        if (!inside && tc < 0) continue;
        cudaCoordinates d = origin + tc * direction - s.c;
        double d2 = pow(cudaMag(d), 2.0);
        if (!inside && pow(s.r, 2.0) < d2) continue;
        double tOffset = sqrt(pow(s.r, 2) - d2) / cudaMag(direction);
        double t = 0.0;
        intersection.found = true;
        t = inside ? tc + tOffset : tc - tOffset;
        if (t < intersection.t) {
            intersection.t = t;
            intersection.c = s.color;
            intersection.center = s.c;
        }
    }
    if (intersection.found == true) 
        intersection.p = intersection.t * direction + origin;
    return intersection;
}

// cuda sphere normal
cudaCoordinates __device__ cudaComputeSphereNormal( cudaCoordinates &p1, cudaCoordinates &c) {
    cudaCoordinates normal = p1 - c;
    cudaNormalize(normal);
    return normal;
}

// cuda check if a point is in shadow
bool __device__ cudaIsInShadow(cudaSphere* spheres, int numSpheres, cudaSun& currentSun, cudaCoordinates& intersection) {
    constexpr double bias = 1e-6;
    cudaCoordinates biasVector = currentSun.direction * bias;
    cudaCoordinates shadowOrigin = intersection + biasVector;
    cudaIntersection i = cudaGetSphereCollision(spheres, numSpheres, shadowOrigin, currentSun.direction);
    return i.found;
}
// cuda color computation
void __device__ cudaComputeColor(cudaSphere* spheres, int numSpheres, cudaSun& currentSun, cudaCoordinates& normal, Color& c, cudaCoordinates& p, cudaCoordinates& eye) {
    if (!cudaIsInShadow(spheres, numSpheres, currentSun, p)) { // check if current sun is null
        cudaCoordinates eyeDir = p - eye;
        cudaNormalize(eyeDir);
        if (cudaDot(eyeDir, normal) > 0.0)
            normal = normal * -1;
        // currentSun->direction.normalize();
        double lambert = max(cudaDot(normal, currentSun.direction), 0.0);
        c.r *= lambert * currentSun.c.r;
        c.g *= lambert * currentSun.c.g;
        c.b *= lambert * currentSun.c.b;
    }
    else {
        c.r = 0.0;
        c.g = 0.0;
        c.b = 0.0;
    }
}

// cuda color pixel at location
void __device__ cudaColorPixel(unsigned char* png, int width, int height, int x, int y, Color &c) {
    cudaConvertLinearTosRGB(c);
    png[((y * width) + x)*4 + 0] = static_cast<unsigned char>(c.r * 255.0);
    png[((y * width) + x)*4 + 1] = static_cast<unsigned char>(c.g * 255.0);
    png[((y * width) + x)*4 + 2] = static_cast<unsigned char>(c.b * 255.0);
    png[((y * width) + x)*4 + 3] = static_cast<unsigned char>(c.alpha * 255.0);
}

/*******END CUDA********/

// printing overload for color
std::ostream& operator<<(std::ostream& out, const Color& c) { out << "(" << c.r << ", " << c.g << ", " << c.b << ')'; return out;}

// Image constructor
Image::Image(int w, int h, std::string n): width{w}, height{h}, name{n} {
    png.resize(height * width * 4);
    maxDim = std::max(width, height);
    currentSun = nullptr;
    eye = new Point;
    forward = new Vector(0, 0, -1);
    right = new Vector(1, 0, 0);
    up = new Vector(0, 1, 0);
}

// Image deconstructor
Image::~Image() {
    delete currentSun;
    delete eye;
    delete forward;
    delete right;
    delete up;
}

// Height getter
int Image::getHeight() {return height;}

// Max Dim getter
double Image::getMaxDim() {return maxDim;}

// sun getter
Sun* Image::getSun() {return currentSun;}

// get eye
Point* Image::getEye() {return eye;}

// get forward
Vector* Image::getForward() {return forward;}

// get right
Vector* Image::getRight() {return right;}

// get up
Vector* Image::getUp() {return up;}

// Width getter
int Image::getWidth() {return width;}

// Coor getter
Color Image::getColor(){return currentColor;}

// Name getter
std::string const &Image::getName() {return name;}

// Png getter
std::vector<unsigned char> const &Image::getPng() {return png;}

// Set the color
void Image::setColor(double r, double g, double b, double a) {
    currentColor.r = clamp(r, 0.0, 1.0);
    currentColor.g = clamp(g, 0.0, 1.0);
    currentColor.b = clamp(b, 0.0, 1.0);
    currentColor.alpha = clamp(a, 0.0, 1.0);
}

// add sphere to list of objects
void Image::addObject(double x, double y, double z, double r) {
    objects.push_back(Sphere{Point{x,y,z}, r, this->currentColor});
}

// Set the current sun
void Image::setSun(double x, double y, double z) {
    if (currentSun == nullptr) {
        currentSun = new Sun;
        currentSun->c = currentColor;
    }
    Vector d(x,y,z);
    d.normalize();
    currentSun->direction = d;
    currentSun->c = currentColor;
}

// Print the current sun
void Image::printSun() {
    if (currentSun != nullptr) {
        std::cout << "Color: " << currentSun->c << std::endl;
        std::cout << "Direction: " << std::endl;
        currentSun->direction.print();
    }
}

// Cast rays and draw the scene
void Image::castRays() {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            double sx = (2.0 * x - width) / maxDim;
            double sy = (height - 2.0 * y) / maxDim;
            Vector direction = *forward + sx * *right + sy * *up;
            direction.normalize();
            const Ray ray = Ray{*eye, direction};
            Intersection intersection = getSphereCollision(ray);
            if (intersection.found == true && intersection.t > 0.0) {
                Vector normal = computeSphereNormal(intersection.p, intersection.center);
                computeColor(normal, intersection.c, intersection.p);
                colorPixel(x, y, intersection.c);
            }
        }
    }
}

// cuda cast rays
__global__ void castRaysKernel(cudaImage* image) {
    const int sharedSpheresCount = 64; // Adjust based on available shared memory
    __shared__ cudaSphere sharedSpheres[sharedSpheresCount];

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    double sx = (2.0 * x - width) / image->maxDim;
    double sy = (height - 2.0 * y) / image->maxDim;
    cudaCoordinates direction = forward + sx * right + sy * up;
    cudaNormalize(direction);

    // cudaCoordinates rayOrigin = eye;
    cudaIntersection closestIntersection;
    closestIntersection.found = false;
    closestIntersection.t = DBL_MAX;

    // Iterate over chunks of spheres
    for (int i = 0; i < numSpheres; i += sharedSpheresCount) {
        // Load a chunk of spheres into shared memory
        int loadIndex = threadIdx.x + threadIdx.y * blockDim.x;
        if (loadIndex < sharedSpheresCount && (i + loadIndex) < numSpheres) {
            sharedSpheres[loadIndex] = image->spheres[i + loadIndex];
        }
        __syncthreads();

        // Check for intersections with spheres in shared memory
        int chunkSize = min(sharedSpheresCount, numSpheres - i);
        cudaIntersection intersection = cudaGetSphereCollision(sharedSpheres, chunkSize, eye, direction);

        // Update closest intersection if needed
        if (intersection.found && intersection.t < closestIntersection.t) {
            closestIntersection = intersection;
        }
        __syncthreads();
    }

    // Use the closest intersection found for coloring the pixel
    if (closestIntersection.found) {
        cudaCoordinates normal = cudaComputeSphereNormal(closestIntersection.p, closestIntersection.center);
        // Note: cudaIsInShadow now checks against the sharedSpheres
        if (!cudaIsInShadow(sharedSpheres, min(sharedSpheresCount, image->numSpheres), constSun, closestIntersection.p)) {
            cudaComputeColor(sharedSpheres, min(sharedSpheresCount, image->numSpheres), constSun, normal, closestIntersection.c, closestIntersection.p, eye);
            cudaColorPixel(image->png, width, height, x, y, closestIntersection.c);
        } else {
            // Handle the case where the point is in shadow
            Color c = {0.0,0.0,0.0,0.0};
            cudaColorPixel(image->png, width, height, x, y, c);
        }
    }
}

void cudaRaytracer(cudaImage *hostImage) {
    const int BLOCK_SIZE = 16;
    cudaSun hostSun;
    int hostHeight = hostImage->height;
    int hostWidth = hostImage->width;
    int hostDim = hostImage->maxDim;
    int hostSpheres = hostImage->numSpheres;
    cudaCoordinates hostEye = hostImage->eye;
    cudaCoordinates hostForward = hostImage->forward;
    cudaCoordinates hostRight = hostImage->right;
    cudaCoordinates hostUp = hostImage->up;
    hostSun.direction = hostImage->currentSun.direction;
    hostSun.c = hostImage->currentSun.c;
    cudaNormalize(hostSun.direction);
    cudaMemcpyToSymbol(constSun, &hostSun, sizeof(cudaSun));
    cudaMemcpyToSymbol(height, &hostHeight, sizeof(int));
    cudaMemcpyToSymbol(maxDim, &hostDim, sizeof(int));
    cudaMemcpyToSymbol(width, &hostWidth, sizeof(int));
    cudaMemcpyToSymbol(numSpheres, &hostSpheres, sizeof(int));
    cudaMemcpyToSymbol(eye, &hostEye, sizeof(cudaCoordinates));
    cudaMemcpyToSymbol(forward, &hostForward, sizeof(cudaCoordinates));
    cudaMemcpyToSymbol(right, &hostRight, sizeof(cudaCoordinates));
    cudaMemcpyToSymbol(up, &hostUp, sizeof(cudaCoordinates));
    // Allocate memory for the cudaImage structure on the device
    cudaImage *deviceImage;
    cudaError_t status = cudaMalloc((void**)&deviceImage, sizeof(cudaImage));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for deviceImage: " << cudaGetErrorString(status) << std::endl;
        return;
    }

    // Copy the cudaImage structure from host to device
    status = cudaMemcpy(deviceImage, hostImage, sizeof(cudaImage), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for deviceImage: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceImage);
        return;
    }

    // Allocate memory for the spheres array on the device
    cudaSphere* deviceSpheres;
    status = cudaMalloc((void**)&deviceSpheres, hostImage->numSpheres * sizeof(cudaSphere));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for deviceSpheres: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceImage);
        return;
    }

    // Copy sphere data from host to device
    status = cudaMemcpy(deviceSpheres, hostImage->spheres, hostImage->numSpheres * sizeof(cudaSphere), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for deviceSpheres: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceSpheres);
        cudaFree(deviceImage);
        return;
    }

    // Update the spheres pointer in the device cudaImage struct
    status = cudaMemcpy(&(deviceImage->spheres), &deviceSpheres, sizeof(cudaSphere*), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy to update deviceImage->spheres failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceSpheres);
        cudaFree(deviceImage);
        return;
    }

    // Allocate memory for the png array on the device
    unsigned char *devicePng;
    status = cudaMalloc((void**)&devicePng, (hostImage->height * hostImage->width * 4) * sizeof(unsigned char));
    if (status != cudaSuccess) {
        std::cerr << "cudaMalloc failed for devicePng: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceSpheres);
        cudaFree(deviceImage);
        return;
    }

    // Copy png data from host to device
    status = cudaMemcpy(devicePng, hostImage->png, (hostImage->height * hostImage->width * 4) * sizeof(unsigned char), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy failed for devicePng: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceSpheres);
        cudaFree(devicePng);
        cudaFree(deviceImage);
        return;
    }

    // Update the png pointer in the device cudaImage struct
    status = cudaMemcpy(&(deviceImage->png), &devicePng, sizeof(unsigned char*), cudaMemcpyHostToDevice);
    if (status != cudaSuccess) {
        std::cerr << "cudaMemcpy to update deviceImage->png failed: " << cudaGetErrorString(status) << std::endl;
        cudaFree(deviceSpheres);
        cudaFree(devicePng);
        cudaFree(deviceImage);
        return;
    }

    // Setup the execution configuration
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((hostImage->width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (hostImage->height + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // Launch the kernel
    castRaysKernel<<<numBlocks, threadsPerBlock>>>(deviceImage);
    
    // Wait for kernel to finish
    cudaDeviceSynchronize();

    // cudaMemcpy(hostImage->spheres, deviceSpheres, hostImage->numSpheres * sizeof(cudaSphere), cudaMemcpyDeviceToHost);
    cudaMemcpy(hostImage->png, devicePng, (hostImage->height * hostImage->width * 4) * sizeof(unsigned char), cudaMemcpyDeviceToHost);
    // Copy the modified cudaImage structure back to the host
    // status = cudaMemcpy(hostImage, deviceImage, sizeof(cudaImage), cudaMemcpyDeviceToHost);
    // if (status != cudaSuccess) {
    //     std::cerr << "cudaMemcpy failed: " << cudaGetErrorString(status) << std::endl;
    // }

    // Free device memory
    cudaFree(deviceSpheres);
    cudaFree(deviceImage);
    cudaFree(devicePng);
}

// return the ray-sphere collision
Intersection Image::getSphereCollision(const Ray &ray) const {
    Intersection intersection;
    for (auto &object : objects) {
        Vector diff(object.c - ray.origin);
        bool inside = std::pow(diff.mag(), 2.0) < std::pow(object.r, 2.0);
        double tc = dot(diff, ray.direction) / ray.direction.mag();
        if (!inside && tc < 0) continue;
        Point d = ray.origin + tc * ray.direction.getVectorAsPoint() - object.c;
        double d2 = std::pow(Vector(d).mag(), 2.0);
        if (!inside && std::pow(object.r, 2.0) < d2) continue;
        double tOffset = std::sqrt(std::pow(object.r, 2) - d2) / ray.direction.mag();
        double t = 0.0;
        intersection.found = true;
        t = inside ? tc + tOffset : tc - tOffset;
        if (t < intersection.t) {
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
void Image::convertLinearTosRGB(Color &c) {
    const double factor = 0.0031308;
    c.r = clamp(c.r, 0.0, 1.0);
    c.g = clamp(c.g, 0.0, 1.0);
    c.b = clamp(c.b, 0.0, 1.0);
    c.r = c.r <= factor ? 12.92 * c.r : 1.055 * std::pow(c.r, (1.0/2.4)) - 0.055;
    c.g = c.g <= factor ? 12.92 * c.g : 1.055 * std::pow(c.g, (1.0/2.4)) - 0.055;
    c.b = c.b <= factor ? 12.92 * c.b: 1.055 * std::pow(c.b, (1.0/2.4)) - 0.055;
}

// compute color with lambert shading
void Image::computeColor(Vector& normal, Color& c, Point& p) {
    if (currentSun != nullptr && !isInShadow(p)) {
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
    else {
        c.r = 0.0;
        c.g = 0.0;
        c.b = 0.0;
    }
}

// print the set of objects in the scene
void Image::printObjects() {
    for (auto &object : objects) {
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
bool Image::isInShadow(const Point &intersection) {
    constexpr double bias = 1e-6;
    Vector biasVector = currentSun->direction * bias;
    Ray shadowRay{intersection + biasVector.getVectorAsPoint(), currentSun->direction};
    Intersection i = getSphereCollision(shadowRay);
    return i.found;
}

// color pixel at location
void Image::colorPixel(int x, int y, Color &c) {
    convertLinearTosRGB(c);
    png[((y * width) + x)*4 + 0] = static_cast<unsigned char>(c.r * 255.0);
    png[((y * width) + x)*4 + 1] = static_cast<unsigned char>(c.g * 255.0);
    png[((y * width) + x)*4 + 2] = static_cast<unsigned char>(c.b * 255.0);
    png[((y * width) + x)*4 + 3] = static_cast<unsigned char>(c.alpha * 255.0);
}

// convert image class to cuda compatible struct 
void convertImageToCudaImage(Image &i, cudaImage &ci) {
    ci.width = i.getWidth();
    ci.height = i.getHeight();
    ci.maxDim = i.getMaxDim();
    ci.currentColor = i.getColor();
    Sun* s = i.getSun();
    ci.currentSun.c = s->c;
    ci.currentSun.direction = cudaCoordinates{s->direction.getX(), s->direction.getY(), s->direction.getZ()};
    ci.eye = cudaCoordinates{i.getEye()->x, i.getEye()->y, i.getEye()->z};
    ci.forward = cudaCoordinates{i.getForward()->getX(), i.getForward()->getY(), i.getForward()->getZ()};
    ci.right = cudaCoordinates{i.getRight()->getX(), i.getRight()->getY(), i.getRight()->getZ()};
    ci.up = cudaCoordinates{i.getUp()->getX(), i.getUp()->getY(), i.getUp()->getZ()};
    ci.spheres = new cudaSphere[i.objects.size()];
    ci.numSpheres = i.objects.size();
    for (int p = 0; p < i.objects.size(); p++) {
        Sphere s = i.objects.at(p);
        cudaSphere cs = {
            cudaCoordinates{s.c.x, s.c.y, s.c.z},
            s.r,
            s.color
        };
        ci.spheres[p] = cs;
    }
    ci.png = new unsigned char[i.getPng().size()];
    std::copy(i.getPng().begin(), i.getPng().end(), ci.png);
}

