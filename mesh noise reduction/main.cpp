
#include "./helpers/functions.h"

using namespace std;


vector<vector<int>> dirs = {{-1,-1}, {-1,0}, {-1,1}, {0,-1},
                            {0,1}, {1,-1}, {1,0}, {1,1}};
int main() {
    point_cloud* pc = new point_cloud(2592,1944); 
    pc->cloud = pc->readPLYFileWithNormals("./docs/OriginalMesh.ply",2592,1944);

    size_t xs = pc->cloud.size();
    size_t ys = pc->cloud[0].size();
    size_t zs = pc->cloud[0][0].size();

    for(int iteration = 0; iteration < 3; iteration++){
        for(size_t x=0; x<xs; x++){
            for(size_t y=0; y<ys; y++){
                if(isnan(pc->cloud[x][y][0]) && isnan(pc->cloud[x][y][1]) && isnan(pc->cloud[x][y][2])){
                    vector<Vertex> neighbourhood;
                    for(int i=0; i<dirs.size(); i++){
                        int tx = x+dirs[i][0];
                        int ty = y+dirs[i][1];
                        if(((tx >= 0) && (tx < xs)) && ((ty >= 0) && (ty < ys)) && !isnan(pc->cloud[tx][ty][0])){
                            neighbourhood.push_back(Vertex(pc->cloud[tx][ty][0],pc->cloud[tx][ty][1],pc->cloud[tx][ty][2]));
                        }
                    }
                    Vertex v(pc->cloud[x][y][0],pc->cloud[x][y][1],pc->cloud[x][y][2]);
                    double sigmaC = pc->sigma_C(pc,x,y,neighbourhood);
                    double sigmaS = pc->sigma_S(pc,x,y,neighbourhood);
                    v = pc->denoise_point(pc,x,y,sigmaC,sigmaS);
                    pc->cloud[x][y][0] = v.x;
                    pc->cloud[x][y][1] = v.y;
                    pc->cloud[x][y][2] = v.z;
                }
            }
        }
    }
    

    cout<<"Finished. Writing temp.ply will start."<<endl;
    
    pc->writePLYFile("./docs/temp.ply",pc->cloud,false);

    cout<<"Writing has finished."<<endl;
    
    return 0;
}