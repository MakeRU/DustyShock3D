
// Area 
double X_min, X_max, Y_min, Y_max, Z_min, Z_max; 
double Particle_on_length;

const double Pi = 3.14159265;
const double Gam_g = 1.4;

// Particle 
int Maximum_particle, Pr, Pm;
double h;

// Time variables
double tau, Tm, T_end;


// Gas 
double * x_gas, * y_gas, * z_gas, * p_gas, * rho_gas, * Vx_gas, * Vy_gas, * Vz_gas;
int * Ind_gas;
double Speed_of_sound_gas;

int Type_of_state;


// Cells to find particle
int* Ne, * Nc, * Cell, * Nn;
int Cnx, Cny, Cnz, Cl; // Cells length 
double Clx, Cly, Clz, Cmx, Cmy, Cmz, Clh; // 


// Dust

// File
int out_num; // Number of output file

// Other variables

int p, Im, Jm, Km;
int i, j, k;