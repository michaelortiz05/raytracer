#include <fstream>
#include <string>
#include <sstream>
#include <typeinfo>
#include "image.h"
#include "lodepng.h"
#include <chrono>
// Source: https://itecnote.com/tecnote/c-how-to-check-if-string-ends-with-txt/
bool has_suffix(const std::string &str, const std::string &suffix)
{
    return str.size() >= suffix.size() &&
           str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// File IO source: learncpp.com
int main(int argc, char *argv[])
{
    std::ifstream inf{argv[1]};
    std::vector<std::string> line;
    Image *img;

    if (argc != 2)
    {
        std::cerr << "No input file!\n";
        return 1;
    }
    if (!inf)
    {
        std::cerr << "Error opening file!\n";
        return 1;
    }

    while (inf)
    {
        line.clear();
        std::string word;
        std::string strInput;

        std::getline(inf, strInput);
        std::stringstream s(strInput);

        while (s >> word)
            line.push_back(word);
        if (line.empty())
            continue;

        if (line[0] == "png")
        {
            if (line.size() != 4 || typeid(line[3]) != typeid(std::string) || !has_suffix(line[3], ".png"))
                break;

            // revisit this if necessary
            // if (img.name != "") {
            //     unsigned error = lodepng::encode(img.name, img.png, img.width, img.height);
            //     if(error) std::cout << "encoder error " << error << ": "<< lodepng_error_text(error) << std::endl;
            // }
            img = new Image(std::stoi(line[1]), std::stoi(line[2]), line[3]);
            unsigned error = lodepng::encode(img->getName(), img->getPng(), img->getWidth(), img->getHeight());
            if (error)
                std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
        }

        else if (line[0] == "sphere")
        {
            if (line.size() != 5)
                break;

            // Grab sphere values
            double x = std::stof(line[1]);
            double y = std::stof(line[2]);
            double z = std::stof(line[3]);
            double r = std::stof(line[4]);

            // Add sphere
            img->addObject(x, y, z, r);
        }

        else if (line[0] == "color")
        {
            if (line.size() != 4)
                break;
            img->setColor(std::stof(line[1]), std::stof(line[2]), std::stof(line[3]));
        }

        else if (line[0] == "sun")
        {
            if (line.size() != 4)
                break;

            // Grab sun values
            double x = std::stof(line[1]);
            double y = std::stof(line[2]);
            double z = std::stof(line[3]);
            img->setSun(x, y, z);
        }
    }
    // img->printObjects();
    std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();
    img->castRays();
    std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time (serial) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[μs]" << std::endl;
    begin = std::chrono::steady_clock::now();
    img->castRays_parallel();
    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time (openMP) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[μs]" << std::endl;
    begin = std::chrono::steady_clock::now();
    img->castRays_threaded();
    end = std::chrono::steady_clock::now();
    std::cout << "Elapsed time (threaded) = " << std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count() << "[μs]" << std::endl;

    if (img->getName() != "")
    {
        unsigned error = lodepng::encode(img->getName(), img->getPng(), img->getWidth(), img->getHeight());
        if (error)
            std::cout << "encoder error " << error << ": " << lodepng_error_text(error) << std::endl;
    }
    delete img;
}