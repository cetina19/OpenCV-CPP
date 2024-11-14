#pragma once

#include "functions.h"


using namespace std;

vector<vector<vector<float>>> point_cloud::readPLYFileWithNormals(const string& filename, int width, int height) {
    ifstream inFile(filename);
    vector<vector<vector<float>>> organized_point_cloud(width, vector<vector<float>>(height, vector<float>(6, numeric_limits<float>::quiet_NaN())));  // Initialize with NaN values

    string line;

    if (!inFile) {
        cerr << "Unable to open file\n";
        return organized_point_cloud;  // return empty opc
    }

    // Read and skip header lines until we reach end_header
    while (getline(inFile, line)) {
        if (line == "end_header") {
            break;
        }
    }

    int nan = 0;
    int no_nan = 0;
    int current_row = 0, current_col = 0;

    // Read data lines
    while (getline(inFile, line) && current_row < width) {
        istringstream lineStream(line);
        float x, y, z, nx, ny, nz;

        // If parsing succeeds and none of the values are NaN
        if ((lineStream >> x >> y >> z >> nx >> ny >> nz)) {
            organized_point_cloud[current_row][current_col] = { x, y, z, nx, ny, nz };
            no_nan++;
        }
        else {
            nan++;
        }

        current_col++;
        if (current_col >= height) {
            current_col = 0;
            current_row++;
        }
    }

    cout << "Nan: " << nan << " Not Nan: " << no_nan << endl;
    return organized_point_cloud;
}

void point_cloud::writePLYFile(const string& filename, const vector<vector<vector<float>>>& organized_point_cloud, bool organized = false) {
    ofstream plyFile(filename);

    if (!plyFile) {
        cout << "Unable to open file";
        return;
    }

    // Calculate total vertices
    size_t totalVertices = 0;
    if (!organized){
        for (const auto& row : organized_point_cloud) {
            for (const auto& pt : row) {
                if (!isnan(pt[0])) { // assuming if x is NaN, the entire point is invalid
                    totalVertices++;
                }
            }
        }
    } 
    else {
        totalVertices = organized_point_cloud.size() * organized_point_cloud[0].size();
    }
   

    plyFile << "ply\n";
    plyFile << "format ascii 1.0\n";
    plyFile << "element vertex " << totalVertices << "\n";
    plyFile << "property float x\n";
    plyFile << "property float y\n";
    plyFile << "property float z\n";
    plyFile << "property float nx\n";
    plyFile << "property float ny\n";
    plyFile << "property float nz\n";
    plyFile << "element face " << 0 << "\n";
    plyFile << "property list uchar int vertex_indices\n";
    plyFile << "end_header\n";

    for (const auto& row : organized_point_cloud) {
        for (const auto& pt : row) {
            if (organized || !isnan(pt[0])) { // assuming if x is NaN, the entire point is invalid
                plyFile << pt[0] << " " << pt[1] << " " << pt[2] << " "
                    << pt[3] << " " << pt[4] << " " << pt[5] << "\n";
            }
        }
    }
}

point_cloud::point_cloud(int width, int height){
    w = width; h = height;
}

point_cloud::~point_cloud() {
    cloud.clear();
}

double point_cloud::euclidean_distance(const Vertex& v1, const Vertex& v2) {
    double dx = v1.x - v2.x;
    double dy = v1.y - v2.y;
    double dz = v1.z - v2.z;
    return sqrt(dx * dx + dy * dy + dz * dz);
}


// Calculating average distance of the neighbours to get sigma_c
double point_cloud::sigma_C(point_cloud *pc, int x, int y, vector<Vertex> neighborhood) {
    Vertex vertex = {pc->cloud[x][y][0], pc->cloud[x][y][1], pc->cloud[x][y][2]};

    double sumDistance = 0.0f;
    for (const Vertex& q : neighborhood) {
        double t = sqrt(pow(vertex.x - q.x, 2) + pow(vertex.y - q.y, 2) + pow(vertex.z - q.z, 2));
        sumDistance += t;
    }

    double sigma_c = sumDistance / neighborhood.size(); // Average Distance
    return sigma_c;
}

// Calculating standard deviation of the neighbours to get sigma_s
double point_cloud::sigma_S(point_cloud *pc, int x, int y, vector<Vertex> neighborhood) {
    Vertex vertex = {pc->cloud[x][y][0], pc->cloud[x][y][1], pc->cloud[x][y][2]};
    Normal normal = {pc->cloud[x][y][3], pc->cloud[x][y][4], pc->cloud[x][y][5]};

    std::vector<float> differences;
    for (const Vertex& q : neighborhood) {
        float h = normal.nx * (vertex.x - q.x) + normal.ny * (vertex.y - q.y) + normal.nz * (vertex.z - q.z);
        differences.push_back(h);
    }

    double meanDifference = accumulate(differences.begin(), differences.end(), 0.0f) / differences.size();
    double accum = 0.0f;
    for (float diff : differences) {
        accum += (diff - meanDifference) * (diff - meanDifference);
    }

    double sigma_s = sqrt(accum / (differences.size() - 1)); // Standard Deviation
    return sigma_s;
}

// Implementing Algorithm for NAN points
Vertex point_cloud::denoise_point(point_cloud *pc, int x, int y, double sigma_c, double sigma_s) {
    std::vector<Vertex> neighborhood; 
    Normal normal = {pc->cloud[x][y][3], pc->cloud[x][y][4], pc->cloud[x][y][5]};
    Vertex vertex = {pc->cloud[x][y][0], pc->cloud[x][y][1], pc->cloud[x][y][2]};

    double sum = 0.0f;
    double normalizer = 0.0f;
    
    for (Vertex& q : neighborhood) {
        double t = sqrt(pow(vertex.x - q.x, 2) + pow(vertex.y - q.y, 2) + pow(vertex.z - q.z, 2)); // Euclidean Distance
        double h = normal.nx * (vertex.x - q.x) + normal.ny * (vertex.y - q.y) + normal.nz * (vertex.z - q.z); // Projection of v - q

        double wc = exp(-t * t / (2 * sigma_c * sigma_c)); // WC
        double ws = exp(-h * h / (2 * sigma_s * sigma_s)); // WS

        sum += (wc * ws) * h; // Height implemented directly with weights
        normalizer += wc * ws;
    }

    double displacement = sum / normalizer;

    Vertex new_vertex;
    new_vertex.x = vertex.x + normal.nx * displacement;
    new_vertex.y = vertex.y + normal.ny * displacement;
    new_vertex.z = vertex.z + normal.nz * displacement;

    return new_vertex;
}

// In this part "sum += (wc * ws) * h;"
// Better implementation is instead of using h (height difference of vertex based on neighbour) in sum, 
// we can use neighbors' value in the multiplication of the weight.