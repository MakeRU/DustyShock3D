
// Area 
double X_min, X_max, Y_min, Y_max, Z_min, Z_max; 
double Particle_on_length;

const double Pi = 3.14159265;
const double Gam_g = 1.4;
const int Num_of_dust_sort_max = 10;

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
int * Ne_gas, * Nc_gas, * Cell_gas, * Nn_gas;
int* Ne_dust[Num_of_dust_sort_max], * Nc_dust[Num_of_dust_sort_max], * Cell_dust[Num_of_dust_sort_max], * Nn_dust[Num_of_dust_sort_max];
int Cnx, Cny, Cnz, Cl; // Cells length 
double Clx, Cly, Clz, Cmx, Cmy, Cmz, Clh; // 


// Dust

double h_cell;
int Num_dust_sort, Number_of_dust_cell;
double * x_dust[Num_of_dust_sort_max], * y_dust[Num_of_dust_sort_max], * z_dust[Num_of_dust_sort_max], * h_dust[Num_of_dust_sort_max], *mas_dust[Num_of_dust_sort_max], *rho_dust[Num_of_dust_sort_max], * Vx_dust[Num_of_dust_sort_max], 
* Vy_dust[Num_of_dust_sort_max], * Vz_dust[Num_of_dust_sort_max], * Ax_dust[Num_of_dust_sort_max], * Ay_dust[Num_of_dust_sort_max], * Az_dust[Num_of_dust_sort_max],  * t_stop[Num_of_dust_sort_max];
int * ind_dust[Num_of_dust_sort_max], * N_dust[Num_of_dust_sort_max];
double mas_gas_dust[Num_of_dust_sort_max], R_dust[Num_of_dust_sort_max];
double Coeff_h_cell, cell_dust_width;

// Average variable

double * x_dust_cell, * y_dust_cell, * z_dust_cell;
double * v_g_average, * rho_g_average, * e_g_average, * Psi_av_new, * v_av_new, * y_av_new, * y_av, * beta_cell;
int * g_average_count;
double* v_d_average[Num_of_dust_sort_max], * rho_d_average[Num_of_dust_sort_max], * eps_cell[Num_of_dust_sort_max], * d_average_count[Num_of_dust_sort_max], * t_stop_average[Num_of_dust_sort_max],
* x_av_new[Num_of_dust_sort_max], * x_av[Num_of_dust_sort_max], * u_av_new[Num_of_dust_sort_max], * b_cell[Num_of_dust_sort_max];


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
int* dev_Nn_gas; // Номер следующей
int* dev_Nc_gas; // Номер ячейки в которой частица
int* dev_Cell_gas; // Номер ячейки

double* dev_e_temp;

double * dev_x_dust[Num_of_dust_sort_max];
double* dev_y_dust[Num_of_dust_sort_max];
double* dev_z_dust[Num_of_dust_sort_max];
double* dev_Vx_dust[Num_of_dust_sort_max];
double* dev_Vy_dust[Num_of_dust_sort_max];
double* dev_Vz_dust[Num_of_dust_sort_max];
double* dev_Ax_dust[Num_of_dust_sort_max];
double* dev_Ay_dust[Num_of_dust_sort_max];
double* dev_Az_dust[Num_of_dust_sort_max];
double* dev_rho_dust[Num_of_dust_sort_max];
double* dev_mas_dust[Num_of_dust_sort_max];
double* dev_Ind_dust[Num_of_dust_sort_max];
int* dev_Nn_dust[Num_of_dust_sort_max]; // Номер следующей
int* dev_Nc_dust[Num_of_dust_sort_max]; // Номер ячейки в которой частица
int* dev_Cell_dust[Num_of_dust_sort_max]; // Номер ячейки

// GPU variables

double *dev_Gam_g;

// Other variables

int p, Im, Jm, Km;
int i, j, k, l, g;

