
// Area 
double X_min, X_max, Y_min, Y_max, Z_min, Z_max; 
double Particle_on_length;

const double Pi = 3.14159265;
const double Gam_g = 1.4;
//const int Num_of_dust_sort_max = 10;

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
int* Ne_dust_1, * Nc_dust_1, * Cell_dust_1, * Nn_dust_1;
int* Ne_dust_2, * Nc_dust_2, * Cell_dust_2, * Nn_dust_2;
int Cnx, Cny, Cnz, Number_of_fihd_cells; // Cells length 
double Clx, Cly, Clz, Cmx, Cmy, Cmz, Clh; // 


// Dust

/*
double h_cell;
int Num_dust_sort, Number_of_dust_cell;
double * x_dust[Num_of_dust_sort_max], * y_dust[Num_of_dust_sort_max], * z_dust[Num_of_dust_sort_max], * h_dust[Num_of_dust_sort_max], *mas_dust[Num_of_dust_sort_max], *rho_dust[Num_of_dust_sort_max], * Vx_dust[Num_of_dust_sort_max], 
* Vy_dust[Num_of_dust_sort_max], * Vz_dust[Num_of_dust_sort_max], * Ax_dust[Num_of_dust_sort_max], * Ay_dust[Num_of_dust_sort_max], * Az_dust[Num_of_dust_sort_max],  * t_stop[Num_of_dust_sort_max];
int * ind_dust[Num_of_dust_sort_max];
double mas_gas_dust[Num_of_dust_sort_max], R_dust[Num_of_dust_sort_max];
double Coeff_h_cell, cell_dust_width;
*/

double* x_dust_1, * y_dust_1, * z_dust_1, * h_dust_1, * mas_dust_1, * rho_dust_1, * Vx_dust_1, * Vy_dust_1, * Vz_dust_1, * Ax_dust_1, * Ay_dust_1, * Az_dust_1, * t_stop_1;
int* ind_dust_1;
double mas_gas_dust_1, R_dust_1;

double* x_dust_2, * y_dust_2, * z_dust_2, * h_dust_2, * mas_dust_2, * rho_dust_2, * Vx_dust_2, * Vy_dust_2, * Vz_dust_2, * Ax_dust_2, * Ay_dust_2, * Az_dust_2, * t_stop_2;
int* ind_dust_2;
double mas_gas_dust_2, R_dust_2;

// Average variable

int Number_of_average_cell, Number_of_average_cell_x, Number_of_average_cell_y, Number_of_average_cell_z;
double Coeff_h_dust_cell, average_cell_width;
// double * x_dust_cell, * y_dust_cell, * z_dust_cell;
double * Vx_g_average, * Vy_g_average, * Vz_g_average, * rho_g_average, * e_g_average, * v_av_new, * y_av_new_x, * y_av_new_y, * y_av_new_z, * y_av_x, * y_av_y, * y_av_z, * beta_cell;
double* Psi_av_new_x, * Psi_av_new_y, * Psi_av_new_z;
int * g_average_count, * d_average_count_1, *d_average_count_2;
double* Vx_d_average_1, * Vy_d_average_1, * Vz_d_average_1, * rho_d_average_1, * eps_cell_1,  * t_stop_average_1,
* x_av_new_x_1, * x_av_new_y_1, * x_av_new_z_1, * x_av_x_1, * x_av_y_1, * x_av_z_1, * u_av_new_1, * b_cell_1;
double* Vx_d_average_2, * Vy_d_average_2, * Vz_d_average_2, * rho_d_average_2, * eps_cell_2, * t_stop_average_2,
* x_av_new_x_2, * x_av_new_y_2, * x_av_new_z_2, * x_av_x_2, * x_av_y_2, * x_av_z_2, * u_av_new_2, * b_cell_2;
int cell_num;


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
int* dev_ind_gas = 0;
int* dev_Nn_gas; // Номер следующей
int* dev_Nc_gas; // Номер ячейки в которой частица
int* dev_Cell_gas; // Номер ячейки

double* dev_e_temp;

double * dev_x_dust_1;
double* dev_y_dust_1;
double* dev_z_dust_1;
double* dev_Vx_dust_1;
double* dev_Vy_dust_1;
double* dev_Vz_dust_1;
double* dev_Ax_dust_1;
double* dev_Ay_dust_1;
double* dev_Az_dust_1;
double* dev_rho_dust_1;
double* dev_mas_dust_1;
int* dev_ind_dust_1;
double* dev_t_stop_1;
int* dev_Nn_dust_1; // Номер следующей
int* dev_Nc_dust_1; // Номер ячейки в которой частица
int* dev_Cell_dust_1; // Номер ячейки

double* dev_x_dust_2;
double* dev_y_dust_2;
double* dev_z_dust_2;
double* dev_Vx_dust_2;
double* dev_Vy_dust_2;
double* dev_Vz_dust_2;
double* dev_Ax_dust_2;
double* dev_Ay_dust_2;
double* dev_Az_dust_2;
double* dev_rho_dust_2;
double* dev_mas_dust_2;
int* dev_ind_dust_2;
double* dev_t_stop_2;
int* dev_Nn_dust_2; // Номер следующей
int* dev_Nc_dust_2; // Номер ячейки в которой частица
int* dev_Cell_dust_2; // Номер ячейки

double* dev_Psi_av_new_x, * dev_Psi_av_new_y, *dev_Psi_av_new_z;
double* dev_Vx_g_average, * dev_Vy_g_average, * dev_Vz_g_average;
double* dev_Vx_d_average_1, * dev_Vy_d_average_1, * dev_Vz_d_average_1;
double* dev_Vx_d_average_2, * dev_Vy_d_average_2, * dev_Vz_d_average_2;
double* dev_t_stop_average_1, * dev_t_stop_average_2;
double* dev_eps_cell_1, * dev_eps_cell_2;

// GPU variables

double *dev_Gam_g;

// Other variables

int p, Im, Jm, Km;
int i, j, k, l, g;

