
// Area 
double X_min, X_max, Y_min, Y_max, Z_min, Z_max; 
double Particle_on_length;

const double Pi = 3.14159265;
const double Gam_g = 1.4;

// Particle 
int Maximum_particle, Pr, Pm;
double h, dlh;

// Time variables
double tau, Tm, T_end, T_out;


// Gas 
double * x_gas, * y_gas, * z_gas, * p_gas, * rho_gas, * mas_gas, * Vx_gas, * Vy_gas, * Vz_gas, * Ax_gas, * Ay_gas, * Az_gas, * e_gas;
int * Ind_gas;
double Speed_of_sound_gas;

int Type_of_state;

// Artificially viscosity

double alpha, beta, eps;

// Cells to find particle
int * Ne, * Nc, * Cell, * Nn;
int Cnx, Cny, Cnz, Cl; // Cells length 
double Clx, Cly, Clz, Cmx, Cmy, Cmz, Clh; // 


// Dust

double h_cell;

// File
int out_num; // Number of output file

// GPU Array

double* dev_p_gas = 0;
double* dev_rho_gas = 0;
double* dev_mas_gas = 0;
double* dev_Ax_gas = 0;
double* dev_Ay_gas = 0;
double* dev_Az_gas = 0;
double* dev_Vx_gas = 0;
double* dev_Vy_gas = 0;
double* dev_Vz_gas = 0;
double* dev_x_gas = 0;
double* dev_y_gas = 0;
double* dev_z_gas = 0;
double* dev_e_gas = 0;
int* dev_Ind_gas = 0;
int* dev_Nn; // Номер следующей
int* dev_Nc; // Номер ячейки в которой частица
int* dev_Cell; // Номер ячейки

// GPU variables

double *dev_Gam_g;

// Other variables

int p, Im, Jm, Km;
int i, j, k, l, g;

