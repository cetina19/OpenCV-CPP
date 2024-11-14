#pragma once

#include <iostream>
#include <fstream>
#include <vector>
#include <limits>
#include <algorithm>
#include <numeric>
#include <opencv2/opencv.hpp>

using namespace std;

struct Vertex {
    float x, y, z;

    Vertex() : x(0), y(0), z(0) {}
    Vertex(float a, float b, float c) : x(a), y(b), z(c) {}

    /*Vertex operator+(Vertex& other) {
        return {x + other.x, y + other.y, z + other.z};
    }

    Vertex operator-(Vertex& other) {
        return {x - other.x, y - other.y, z - other.z};
    }

    Vertex operator*(float scalar) {
        return {x * scalar, y * scalar, z * scalar};
    }

    Vertex& operator+=(Vertex& other) {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }

    Vertex& operator*=(double scalar) {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vertex cross(const Vertex& other) const {
        return {y * other.z - z * other.y, z * other.x - x * other.z, x * other.y - y * other.x};
    }

    float dot(const Vertex& other) const {
        return x * other.x + y * other.y + z * other.z;
    }

    Vertex normalize() {
        float nx, ny, nz;
        float length = std::sqrt(x * x + y * y + z * z);
        if (length > 0) {
            nx = x / length;
            ny = y / length;
            nz = z / length;
        }
        else{
            nx = 0;
            ny = 0;
            nz = 0;
        }
        
        return Vertex(nx,ny,nz);
    }*/
};

struct Normal {
    float nx, ny, nz;

    Normal() : nx(0), ny(0), nz(0) {}
    Normal(float a, float b, float c) : nx(a), ny(b), nz(c) {}
    Normal(Vertex v) : nx(v.x), ny(v.y), nz(v.z) {} 

    Vertex operator*(float scalar) const {
        return {nx * scalar, ny * scalar, nz * scalar};
    }

    float dot(const Vertex& v) const {
        return nx * v.x + ny * v.y + nz * v.z;
    }
};

class point_cloud{
    public:
        int size;
        int w;
        int h;
        int channels = 3;
        vector<vector<vector<float>>> cloud;
        
        point_cloud(){ w = 2592; h = 1944; };
        point_cloud(int w, int h);

        vector<vector<vector<float>>> readPLYFileWithNormals(const string& filename, int width, int height);
        void writePLYFile(const string& filename, const vector<vector<vector<float>>>& organized_point_cloud, bool organized);
        
        double euclidean_distance(const Vertex& v1, const Vertex& v2);
        double sigma_C(point_cloud *pc, int x, int y, vector<Vertex> neighborhood);
        double sigma_S(point_cloud *pc, int x, int y, vector<Vertex> neighborhood);
        Vertex denoise_point(point_cloud *pc, int x, int y, double sigma_c, double sigma_s);

        ~point_cloud();
};