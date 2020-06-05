
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <math.h>
#include <time.h>

#include "data.h"


__device__ __host__ double W(double r, double h)
{
	const double Pi = 3.14159265;
	double k, c, tmp1;
	k = (double)fabs(r) / h;
	c = (double)1 / (Pi);
	if (k < 1.0) { tmp1 = (double) 1.0 - 1.5 * k * k + 0.75 * k * k * k; }
	if ((k >= 1.0) && (k <= 2.0)) { tmp1 = (double) 0.25 * (2.0 - k) * (2.0 - k) * (2.0 - k); }
	if (k > 2.0) { tmp1 = 0.0; }

	return  c / (h * h * h) * tmp1;
}

__device__ __host__ double dW(double r, double h)
{
	const double Pi = 3.14159265;
	double k, c, tmp1;
	k = (double) r / h;
	c = (double) 1 / (Pi);
	if (k < -2.0) { tmp1 = 0.0; }
	if ((k >= -2.0) && (k <= -1.0)) { tmp1 = (double) 0.75 * (2.0 + k) * (2.0 + k); }
	if ((k > -1.0) && (k < 0)) { tmp1 = (double) -3.0 * k - 2.25 * k * k; }
	if ((k >= 0) && (k <= 1.0)) { tmp1 = (double) -3.0 * k + 2.25 * k * k; }
	if ((k >= 1.0) && (k <= 2.0)) { tmp1 = (double) -0.75 * (2.0 - k) * (2.0 - k); }
	if (k > 2.0) { tmp1 = 0.0; }

	return  c / (h * h * h * h) * tmp1;
}

#include "IDIC.h"

__device__ __host__ double Eq_t_stop(double R_dust, double p, double rho, double gamma)
{
	return R_dust / (sqrt(gamma * p / rho) * rho); 
}

__global__ void RhoInitKernel(double* rho, int* Ind, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		rho[i] = 0.0;
	}
}

__global__ void RhoKernel(double* x, double* y, double* z, double* rho, double* mas, int* Ind, int* Cell, int* Nn, int Pm, double Clx, double Cly, double Clz, double Clh, int Cnx, int Cny, int Cl, double h0, int k, int l, int g)
{

	int j, Ni, Ci;
	double d, rho_tmp;

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		Ni = int((x[i] - Clx) / Clh) + (int((y[i] - Cly) / Clh)) * Cnx + (int((z[i] - Clz) / Clh)) * Cnx * Cny;
		rho_tmp = 0.0;
		Ci = Ni + k + l * Cnx + g * Cnx * Cny;
		if ((Ci > -1) && (Ci < Cl))
		{
			j = Cell[Ci];

			while (j > -1)
			{
				d = pow((x[j] - x[i]) * (x[j] - x[i]) + (y[j] - y[i]) * (y[j] - y[i]) + (z[j] - z[i]) * (z[j] - z[i]), 0.5);
				rho_tmp = rho_tmp + mas[j] * W(d, h0);
				j = Nn[j];
			}
		}

		rho[i] = rho[i] + rho_tmp;
	}
}



__device__ __host__ double Eq_State(double rho, double e, int Type_of_state, double Gam_g, double Speed_of_sound_gas)
{
	//	return B*(pow(rho,Gam_liq)-1) + 1;
	if (Type_of_state == 0) { return rho * e * (Gam_g - 1); }
	if (Type_of_state == 1) { return Speed_of_sound_gas * Speed_of_sound_gas * rho; }
}

__device__ __host__ double Eq_State1(double p, double e, int Type_of_state, double Gam_g, double Speed_of_sound_gas)
{
	//	return pow((p-1)/B+1,1.0/Gam_liq);

	if (Type_of_state == 0) { return p / (e * (Gam_g - 1)); }
	if (Type_of_state == 1) { return p / (Speed_of_sound_gas * Speed_of_sound_gas); }
}

__global__ void PKernel(double* rho, double* p, double* e, int* Ind, int Pm, int Type_of_state, double Gam_g, double Speed_of_sound_gas)
{

	int k, l, g, j, Ni, Ci;
	double d, rho_tmp;

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		p[i] = Eq_State(rho[i], e[i], Type_of_state, Gam_g, Speed_of_sound_gas);
		
	}


}


__global__ void ForceInitKernel(double* Ax, double* Ay, double* Az, int* Ind, double* e_temp, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0 )
	{
		Ax[i] = 0.0;
		Ay[i] = 0.0;
		Az[i] = 0.0;
		e_temp[i] = 0.0;
	}
}


__global__ void ForceKernel(double* x, double* y, double* z, double* rho_gas, double* p_gas, double* mas_gas, double* e, double* Vx, double* Vy, double* Vz, double* Ax, double* Ay, double* Az, double* e_temp, int* Ind, int* Cell, int* Nn, int Pm, double Clx, double Cly, double Clz, double Clh, int Cnx, int Cny, int Cl, double h, double tau, double Gam_g, double alpha, double beta, double eps, int k, int l, int g)
{

	int j, Ni, Ci;
	double d, F_nu, nu1x, nu1y, nu1z, nu_temp, dist, A, F_tens, Cnu;

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;
	if (i > Pm) return;

	if (Ind[i] == 0)
	{

		Ni = int((x[i] - Clx) / Clh) + (int((y[i] - Cly) / Clh)) * Cnx + (int((z[i] - Clz) / Clh)) * Cnx * Cny;
		Ci = Ni + k + l * Cnx + g * Cnx * Cny;
		if ((Ci > -1) && (Ci < Cl))
		{
			j = Cell[Ci];

			while (j > -1)
			{
				dist = pow((x[j] - x[i]) * (x[j] - x[i]) + (y[j] - y[i]) * (y[j] - y[i]) + (z[j] - z[i]) * (z[j] - z[i]), 0.5);
				if ((dist < 2 * h) && (dist > 0))
				{

					// Исск. вязкость

					F_nu = 0;
					nu1x = (Vx[i] - Vx[j]) * (x[i] - x[j]);
					nu1y = (Vy[i] - Vy[j]) * (y[i] - y[j]);
					nu1z = (Vz[i] - Vz[j]) * (z[i] - z[j]);
					nu_temp = nu1x + nu1y + nu1z;
					if (nu_temp < 0.0)
					{
						Cnu = 0.5 * (sqrt(Gam_g * p_gas[i] / rho_gas[i]) + sqrt(Gam_g * p_gas[j] / rho_gas[j]));
						nu_temp = h * nu_temp / (dist *dist + eps * h * h);
						F_nu = (-alpha * Cnu * nu_temp + beta * nu_temp * nu_temp) / (0.5 * (rho_gas[i] + rho_gas[j]));
					}

				//	F_tens = kapp * mas[j] * dist * W(dist, h0);
					//e_temp = e_temp + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i]) + mas_gas[j] * p_gas[j] / (rho_gas[j] * rho_gas[j]) + mas_gas[j] / 2.0 * F_nu) * 
					//	((Vx[i] - Vx[j]) * dW(x[i] - x[j], h) + (Vy[i] - Vy[j]) * dW(y[i] - y[j], h) + (Vz[i] - Vz[j]) * dW(z[i] - z[j], h));
					// e_temp = e_temp + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i]) + mas_gas[j] * p_gas[j] / (rho_gas[j] * rho_gas[j]) + mas_gas[j] / 2.0 * F_nu) *
					//	((Vx[i] - Vx[j]) * dW(x[i] - x[j], h) + (Vy[i] - Vy[j]) * dW(y[i] - y[j], h) + (Vz[i] - Vz[j]) * dW(z[i] - z[j], h));
					e_temp[i] = e_temp[i] + (mas_gas[i] * p_gas[i] / (rho_gas[i] * rho_gas[i]) + mas_gas[i] * F_nu / 2.0) *
						((Vx[i] - Vx[j]) * dW(x[i] - x[j], h) + (Vy[i] - Vy[j]) * dW(y[i] - y[j], h) + (Vz[i] - Vz[j]) * dW(z[i] - z[j], h));
					A = mas_gas[j] * (p_gas[i] / (rho_gas[i] * rho_gas[i]) + p_gas[j] / (rho_gas[j] * rho_gas[j]) + F_nu) * dW(dist, h);
					d = (x[j] - x[i]);
					Ax[i] = Ax[i] + d * A / dist;
					d = (y[j] - y[i]);
					Ay[i] = Ay[i] + d * A / dist;
					d = (z[j] - z[i]);
					Az[i] = Az[i] + d * A / dist;
				}



				j = Nn[j];
			}
		}

	}
}

__global__ void MoveKernel(double* x, double* y, double* z, double* Vx, double* Vy, double* Vz, double* Ax, double* Ay, double* Az, int* Ind, double* e_temp, double* e_gas, double tau, int Pm)
{

	int i = blockIdx.x * 256 + threadIdx.x + threadIdx.y * 16;

	if (i > Pm) return;

	if (Ind[i] == 0)
	{
		Vx[i] = Vx[i] + Ax[i] * tau;
		Vy[i] = Vy[i] + Ay[i] * tau;
		Vz[i] = Vz[i] + Az[i] * tau;
		e_gas[i] = e_gas[i] + e_temp[i] * tau;
 


		x[i] = x[i] + Vx[i] * tau;
		y[i] = y[i] + Vy[i]*tau;
		z[i] = z[i] + Vz[i]*tau;


	}

}



void Data_out(int num)
{
	FILE* out_file_gas, * out_file_dust, * out_file;
	char out_name[25];
	int i, j, l;
	double r,v, eps_out;


	eps_out = 1.0*h;

	if (num < 10000000) { sprintf(out_name, "Data/G%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/G0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/G00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/G000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/G0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/G00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/G000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
	//	if ((x_gas[i] >= 0.0) && (x_gas[i] <= 1.0))
		{
			r = sqrt(x_gas[i]* x_gas[i] + y_gas[i]*y_gas[i]+ z_gas[i]*z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, e_gas[i], Ind_gas[i], Nn_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_X%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_X0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_X00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_X000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_X0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_X00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_X000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if ((y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i] <= eps_out * eps_out))
		{
			r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	/*
	if (num < 10000000) { sprintf(out_name, "Data/L_Y%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_Y0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_Y00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_Y000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_Y0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_Y00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_Y000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if ((x_gas[i] * x_gas[i] + z_gas[i] * z_gas[i] <= eps_out * eps_out))
		{
			r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_Z%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_Z0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_Z00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_Z000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_Z0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_Z00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_Z000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if ((x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] <= eps_out * eps_out))
		{
			r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_XY%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_XY0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_XY00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_XY000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_XY0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_XY00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_XY000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if (abs(z_gas[i]) <= eps_out)
		{
			r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
			v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_XZ%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_XZ0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_XZ00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_XZ000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_XZ0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_XZ00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_XZ000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if (abs(y_gas[i]) <= eps_out)
		{
		r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
		v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
		fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
			x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

	if (num < 10000000) { sprintf(out_name, "Data/L_YZ%d.dat", num); };
	if (num < 1000000) { sprintf(out_name, "Data/L_YZ0%d.dat", num); };
	if (num < 100000) { sprintf(out_name, "Data/L_YZ00%d.dat", num); };
	if (num < 10000) { sprintf(out_name, "Data/L_YZ000%d.dat", num); };
	if (num < 1000) { sprintf(out_name, "Data/L_YZ0000%d.dat", num); };
	if (num < 100) { sprintf(out_name, "Data/L_YZ00000%d.dat", num); };
	if (num < 10) { sprintf(out_name, "Data/L_YZ000000%d.dat", num); };

	out_file_gas = fopen(out_name, "wt");
	fprintf(out_file_gas, "t=%5.3f \n", Tm);
	fprintf(out_file_gas, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
	fprintf(out_file_gas, "x \t y \t z \t r \t mas \t rho \t p \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if (abs(x_gas[i]) <= eps_out)
		{
		r = sqrt(x_gas[i] * x_gas[i] + y_gas[i] * y_gas[i] + z_gas[i] * z_gas[i]);
		v = sqrt(Vx_gas[i] * Vx_gas[i] + Vy_gas[i] * Vy_gas[i] + Vz_gas[i] * Vz_gas[i]);
		fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
			x_gas[i], y_gas[i], z_gas[i], r, mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], v, Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);
	*/
}

void Data_out_dust(int num)
{
	FILE * out_file_dust, * out_file;
	char out_name[25];
	int i, j, l, dust_id;
	double r, v, eps_out;


	eps_out = 1.0 * h;

		if (num < 10000000) { sprintf(out_name, "Data/D-1-%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-1-0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-1-00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-1-000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-1-0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-1-00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-1-000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t t_stop \t Vx \t Vy \t Vz \t V \t Ind \t \Nn \n");

		for (i = 0; i <= Pm; i++)
		{
			//	if ((x_dust_1[i] >= 0.0) && (x_dust_1[i] <= 1.0))
			{
				r = sqrt(x_dust_1[i] * x_dust_1[i] + y_dust_1[i] * y_dust_1[i] + z_dust_1[i] * z_dust_1[i]);
				v = sqrt(Vx_dust_1[i] * Vx_dust_1[i] + Vy_dust_1[i] * Vy_dust_1[i] + Vz_dust_1[i] * Vz_dust_1[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \t %d \n",
					x_dust_1[i], y_dust_1[i], z_dust_1[i], r, mas_dust_1[i], rho_dust_1[i], t_stop_1[i], Vx_dust_1[i], Vy_dust_1[i], Vy_dust_1[i], v, ind_dust_1[i], Nn_dust_1[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-1-L_X%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-1-L_X0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-1-L_X00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-1-L_X000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-1-L_X0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-1-L_X00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-1-L_X000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");

		for (i = 0; i <= Pm; i++)
		{
			if ((y_dust_1[i] * y_dust_1[i] + z_dust_1[i] * z_dust_1[i] <= eps_out * eps_out))
			{
				r = sqrt(x_dust_1[i] * x_dust_1[i] + y_dust_1[i] * y_dust_1[i] + z_dust_1[i] * z_dust_1[i]);
				v = sqrt(Vx_dust_1[i] * Vx_dust_1[i] + Vy_dust_1[i] * Vy_dust_1[i] + Vz_dust_1[i] * Vz_dust_1[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_1[i], y_dust_1[i], z_dust_1[i], r, mas_dust_1[i], rho_dust_1[i], Vx_dust_1[i], Vy_dust_1[i], Vy_dust_1[i], v, Ax_dust_1[i], Ay_dust_1[i], Az_dust_1[i], t_stop_1[i], ind_dust_1[i]);
			}

		}

		fclose(out_file_dust);
		/*
		if (num < 10000000) { sprintf(out_name, "Data/D-1-L_Y%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-1-L_Y0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-1-L_Y00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-1-L_Y000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-1-L_Y0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-1-L_Y00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-1-L_Y000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");

		for (i = 0; i <= Pm; i++)
		{
			if ((x_dust_1[i] * x_dust_1[i] + z_dust_1[i] * z_dust_1[i] <= eps_out * eps_out))
			{
				r = sqrt(x_dust_1[i] * x_dust_1[i] + y_dust_1[i] * y_dust_1[i] + z_dust_1[i] * z_dust_1[i]);
				v = sqrt(Vx_dust_1[i] * Vx_dust_1[i] + Vy_dust_1[i] * Vy_dust_1[i] + Vz_dust_1[i] * Vz_dust_1[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_1[i], y_dust_1[i], z_dust_1[i], r, mas_dust_1[i], rho_dust_1[i], Vx_dust_1[i], Vy_dust_1[i], Vy_dust_1[i], v, Ax_dust_1[i], Ay_dust_1[i], Az_dust_1[i], ind_dust_1[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-1-L_Z%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-1-L_Z0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-1-L_Z00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-1-L_Z000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-1-L_Z0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-1-L_Z00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-1-L_Z000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");

		for (i = 0; i <= Pm; i++)
		{
			if ((x_dust_1[i] * x_dust_1[i] + y_dust_1[i] * y_dust_1[i] <= eps_out * eps_out))
			{
				r = sqrt(x_dust_1[i] * x_dust_1[i] + y_dust_1[i] * y_dust_1[i] + z_dust_1[i] * z_dust_1[i]);
				v = sqrt(Vx_dust_1[i] * Vx_dust_1[i] + Vy_dust_1[i] * Vy_dust_1[i] + Vz_dust_1[i] * Vz_dust_1[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_1[i], y_dust_1[i], z_dust_1[i], r, mas_dust_1[i], rho_dust_1[i], Vx_dust_1[i], Vy_dust_1[i], Vy_dust_1[i], v, Ax_dust_1[i], Ay_dust_1[i], Az_dust_1[i], ind_dust_1[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-1-L_XY%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-1-L_XY0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-1-L_XY00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-1-L_XY000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-1-L_XY0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-1-L_XY00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-1-L_XY000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n"); 
		

		for (i = 0; i <= Pm; i++)
		{
			if (abs(z_dust_1[i]) <= eps_out)
			{
				r = sqrt(x_dust_1[i] * x_dust_1[i] + y_dust_1[i] * y_dust_1[i] + z_dust_1[i] * z_dust_1[i]);
				v = sqrt(Vx_dust_1[i] * Vx_dust_1[i] + Vy_dust_1[i] * Vy_dust_1[i] + Vz_dust_1[i] * Vz_dust_1[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_1[i], y_dust_1[i], z_dust_1[i], r, mas_dust_1[i], rho_dust_1[i], Vx_dust_1[i], Vy_dust_1[i], Vy_dust_1[i], v, Ax_dust_1[i], Ay_dust_1[i], Az_dust_1[i], ind_dust_1[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-1-L_XZ%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-1-L_XZ0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-1-L_XZ00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-1-L_XZ000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-1-L_XZ0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-1-L_XZ00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-1-L_XZ000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");
		

		for (i = 0; i <= Pm; i++)
		{
			if (abs(y_dust_1[i]) <= eps_out)
			{
				r = sqrt(x_dust_1[i] * x_dust_1[i] + y_dust_1[i] * y_dust_1[i] + z_dust_1[i] * z_dust_1[i]);
				v = sqrt(Vx_dust_1[i] * Vx_dust_1[i] + Vy_dust_1[i] * Vy_dust_1[i] + Vz_dust_1[i] * Vz_dust_1[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_1[i], y_dust_1[i], z_dust_1[i], r, mas_dust_1[i], rho_dust_1[i], Vx_dust_1[i], Vy_dust_1[i], Vy_dust_1[i], v, Ax_dust_1[i], Ay_dust_1[i], Az_dust_1[i], ind_dust_1[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-1-L_YZ%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-1-L_YZ0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-1-L_YZ00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-1-L_YZ000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-1-L_YZ0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-1-L_YZ00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-1-L_YZ000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");

		for (i = 0; i <= Pm; i++)
		{
			if (abs(x_dust_1[i]) <= eps_out)
			{
				r = sqrt(x_dust_1[i] * x_dust_1[i] + y_dust_1[i] * y_dust_1[i] + z_dust_1[i] * z_dust_1[i]);
				v = sqrt(Vx_dust_1[i] * Vx_dust_1[i] + Vy_dust_1[i] * Vy_dust_1[i] + Vz_dust_1[i] * Vz_dust_1[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_1[i], y_dust_1[i], z_dust_1[i], r, mas_dust_1[i], rho_dust_1[i], Vx_dust_1[i], Vy_dust_1[i], Vy_dust_1[i], v, Ax_dust_1[i], Ay_dust_1[i], Az_dust_1[i], ind_dust_1[i]);
			}

		}

		fclose(out_file_dust);
*/	
		if (num < 10000000) { sprintf(out_name, "Data/D-2-%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-2-0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-2-00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-2-000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-2-0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-2-00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-2-000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t t_stop \t Vx \t Vy \t Vz \t V \t Ind \t Nn \n");

		for (i = 0; i <= Pm; i++)
		{
			//	if ((x_dust_2[i] >= 0.0) && (x_dust_2[i] <= 1.0))
			{
				r = sqrt(x_dust_2[i] * x_dust_2[i] + y_dust_2[i] * y_dust_2[i] + z_dust_2[i] * z_dust_2[i]);
				v = sqrt(Vx_dust_2[i] * Vx_dust_2[i] + Vy_dust_2[i] * Vy_dust_2[i] + Vz_dust_2[i] * Vz_dust_2[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \t %d \n",
					x_dust_2[i], y_dust_2[i], z_dust_2[i], r, mas_dust_2[i], rho_dust_2[i], t_stop_2[i], Vx_dust_2[i], Vy_dust_2[i], Vy_dust_2[i], v, ind_dust_2[i], Nn_dust_2[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-2-L_X%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-2-L_X0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-2-L_X00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-2-L_X000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-2-L_X0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-2-L_X00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-2-L_X000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");

		for (i = 0; i <= Pm; i++)
		{
			if ((y_dust_2[i] * y_dust_2[i] + z_dust_2[i] * z_dust_2[i] <= eps_out * eps_out))
			{
				r = sqrt(x_dust_2[i] * x_dust_2[i] + y_dust_2[i] * y_dust_2[i] + z_dust_2[i] * z_dust_2[i]);
				v = sqrt(Vx_dust_2[i] * Vx_dust_2[i] + Vy_dust_2[i] * Vy_dust_2[i] + Vz_dust_2[i] * Vz_dust_2[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_2[i], y_dust_2[i], z_dust_2[i], r, mas_dust_2[i], rho_dust_2[i], Vx_dust_2[i], Vy_dust_2[i], Vy_dust_2[i], v, Ax_dust_2[i], Ay_dust_2[i], Az_dust_2[i], t_stop_2[i], ind_dust_2[i]);
			}

		}

		fclose(out_file_dust);
		/*
		if (num < 10000000) { sprintf(out_name, "Data/D-2-L_Y%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-2-L_Y0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-2-L_Y00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-2-L_Y000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-2-L_Y0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-2-L_Y00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-2-L_Y000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");

		for (i = 0; i <= Pm; i++)
		{
			if ((x_dust_2[i] * x_dust_2[i] + z_dust_2[i] * z_dust_2[i] <= eps_out * eps_out))
			{
				r = sqrt(x_dust_2[i] * x_dust_2[i] + y_dust_2[i] * y_dust_2[i] + z_dust_2[i] * z_dust_2[i]);
				v = sqrt(Vx_dust_2[i] * Vx_dust_2[i] + Vy_dust_2[i] * Vy_dust_2[i] + Vz_dust_2[i] * Vz_dust_2[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_2[i], y_dust_2[i], z_dust_2[i], r, mas_dust_2[i], rho_dust_2[i], Vx_dust_2[i], Vy_dust_2[i], Vy_dust_2[i], v, Ax_dust_2[i], Ay_dust_2[i], Az_dust_2[i], ind_dust_2[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-2-L_Z%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-2-L_Z0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-2-L_Z00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-2-L_Z000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-2-L_Z0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-2-L_Z00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-2-L_Z000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");

		for (i = 0; i <= Pm; i++)
		{
			if ((x_dust_2[i] * x_dust_2[i] + y_dust_2[i] * y_dust_2[i] <= eps_out * eps_out))
			{
				r = sqrt(x_dust_2[i] * x_dust_2[i] + y_dust_2[i] * y_dust_2[i] + z_dust_2[i] * z_dust_2[i]);
				v = sqrt(Vx_dust_2[i] * Vx_dust_2[i] + Vy_dust_2[i] * Vy_dust_2[i] + Vz_dust_2[i] * Vz_dust_2[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_2[i], y_dust_2[i], z_dust_2[i], r, mas_dust_2[i], rho_dust_2[i], Vx_dust_2[i], Vy_dust_2[i], Vy_dust_2[i], v, Ax_dust_2[i], Ay_dust_2[i], Az_dust_2[i], ind_dust_2[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-2-L_XY%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-2-L_XY0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-2-L_XY00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-2-L_XY000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-2-L_XY0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-2-L_XY00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-2-L_XY000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");


		for (i = 0; i <= Pm; i++)
		{
			if (abs(z_dust_2[i]) <= eps_out)
			{
				r = sqrt(x_dust_2[i] * x_dust_2[i] + y_dust_2[i] * y_dust_2[i] + z_dust_2[i] * z_dust_2[i]);
				v = sqrt(Vx_dust_2[i] * Vx_dust_2[i] + Vy_dust_2[i] * Vy_dust_2[i] + Vz_dust_2[i] * Vz_dust_2[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_2[i], y_dust_2[i], z_dust_2[i], r, mas_dust_2[i], rho_dust_2[i], Vx_dust_2[i], Vy_dust_2[i], Vy_dust_2[i], v, Ax_dust_2[i], Ay_dust_2[i], Az_dust_2[i], ind_dust_2[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-2-L_XZ%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-2-L_XZ0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-2-L_XZ00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-2-L_XZ000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-2-L_XZ0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-2-L_XZ00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-2-L_XZ000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");


		for (i = 0; i <= Pm; i++)
		{
			if (abs(y_dust_2[i]) <= eps_out)
			{
				r = sqrt(x_dust_2[i] * x_dust_2[i] + y_dust_2[i] * y_dust_2[i] + z_dust_2[i] * z_dust_2[i]);
				v = sqrt(Vx_dust_2[i] * Vx_dust_2[i] + Vy_dust_2[i] * Vy_dust_2[i] + Vz_dust_2[i] * Vz_dust_2[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_2[i], y_dust_2[i], z_dust_2[i], r, mas_dust_2[i], rho_dust_2[i], Vx_dust_2[i], Vy_dust_2[i], Vy_dust_2[i], v, Ax_dust_2[i], Ay_dust_2[i], Az_dust_2[i], ind_dust_2[i]);
			}

		}

		fclose(out_file_dust);

		if (num < 10000000) { sprintf(out_name, "Data/D-2-L_YZ%d.dat", num); };
		if (num < 1000000) { sprintf(out_name, "Data/D-2-L_YZ0%d.dat", num); };
		if (num < 100000) { sprintf(out_name, "Data/D-2-L_YZ00%d.dat", num); };
		if (num < 10000) { sprintf(out_name, "Data/D-2-L_YZ000%d.dat", num); };
		if (num < 1000) { sprintf(out_name, "Data/D-2-L_YZ0000%d.dat", num); };
		if (num < 100) { sprintf(out_name, "Data/D-2-L_YZ00000%d.dat", num); };
		if (num < 10) { sprintf(out_name, "Data/D-2-L_YZ000000%d.dat", num); };

		out_file_dust = fopen(out_name, "wt");
		fprintf(out_file_dust, "t=%5.3f \n", Tm);
		fprintf(out_file_dust, "tau=%10.8lf \t h=%10.8lf \n", tau, h);
		fprintf(out_file_dust, "x \t y \t z \t r \t mas \t rho \t Vx \t Vy \t Vz \t V \t Ax \t Ay \t Az \t Ind \n");

		for (i = 0; i <= Pm; i++)
		{
			if (abs(x_dust_2[i]) <= eps_out)
			{
				r = sqrt(x_dust_2[i] * x_dust_2[i] + y_dust_2[i] * y_dust_2[i] + z_dust_2[i] * z_dust_2[i]);
				v = sqrt(Vx_dust_2[i] * Vx_dust_2[i] + Vy_dust_2[i] * Vy_dust_2[i] + Vz_dust_2[i] * Vz_dust_2[i]);
				fprintf(out_file_dust, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
					x_dust_2[i], y_dust_2[i], z_dust_2[i], r, mas_dust_2[i], rho_dust_2[i], Vx_dust_2[i], Vy_dust_2[i], Vy_dust_2[i], v, Ax_dust_2[i], Ay_dust_2[i], Az_dust_2[i], ind_dust_2[i]);
			}

		}

		fclose(out_file_dust);
*/
}


void init_Ball()
{
	 double x_temp, y_temp, z_temp;
	 
	p = -1;
	Im = int((X_max - X_min) * Particle_on_length);
	Jm = int((Y_max - Y_min) * Particle_on_length);
	Km = int((Z_max - Z_min) * Particle_on_length);

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = X_min + (double)(i * dlh);
				y_temp = Y_min + (double)(j * dlh);
				z_temp = Z_min + (double)(k * dlh);
				if (x_temp * x_temp + y_temp * y_temp + z_temp * z_temp <= 1.0) {
				p = p + 1;
				x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
				y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
				z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
				mas_gas[p] = 1.0 / (Particle_on_length * Particle_on_length * Particle_on_length);
				Vx_gas[p] = 0.0;
				Vy_gas[p] = 0.0;
				Vz_gas[p] = 0.0;
				Ax_gas[p] = 0.0;
				Ay_gas[p] = 0.0;
				Az_gas[p] = 0.0;
				e_gas[p] = 2.5;
				p_gas[p] = 1.0;
				rho_gas[p] = 1.0;
				Ind_gas[p] = 0;
				};
			}

	Pr = p;

	Pm = p;
}

void init_Sod_X()
{
	double x_temp, y_temp, z_temp;
	double p_left, p_right, e_left, e_right, rho_left, rho_right;
	
	p = -1;
	Im = int((X_max - X_min) * Particle_on_length);
	Jm = int((Y_max - Y_min) * Particle_on_length);
	Km = int((Z_max - Z_min) * Particle_on_length);

	p_left = 1.0;
	p_right = 0.1;
	e_left = 2.5;
	e_right = 2.0;
	rho_left = Eq_State1(p_left, e_left, Type_of_state, 1.4, 1.0);
	rho_right = Eq_State1(p_right, e_right, Type_of_state, 1.4, 1.0);
	

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = X_min + (double)(i * dlh);
				y_temp = Y_min + (double)(j * dlh);
				z_temp = Z_min + (double)(k * dlh);
				if ((abs(x_temp) <= 1.0) && (abs(y_temp) <= 1.0) && (abs(z_temp) <= 1.0))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					if (x_temp <= 0.0)
					{
						e_gas[p] = e_left;
						p_gas[p] = p_left;
						rho_gas[p] = rho_left;
						mas_gas[p] = rho_left / (Particle_on_length * Particle_on_length * Particle_on_length);
					}
					else
					{
						e_gas[p] = e_right;
						p_gas[p] = p_right;
						rho_gas[p] = rho_right;
						mas_gas[p] = rho_right / (Particle_on_length * Particle_on_length * Particle_on_length);
					}
					if ((abs(x_temp) <= 0.8) && (abs(y_temp) <= 0.8) && (abs(z_temp) <= 0.8))
					{
						Ind_gas[p] = 0;
					}
					else
					{
						Ind_gas[p] = 1;
					}
					

				};
			}

	Pr = p;

	Pm = p;
}

void init_Sod_X_mas()
{
	double x_temp, y_temp, z_temp;
	double p_left, p_right, e_left, e_right, rho_left, rho_right, dlh_left, dlh_right, mas_particle;
	
	double border_length = 2.5 * h;


	p_left = 1.0;
	p_right = 0.1;
	e_left = 2.5;
	e_right = 2.0;
	rho_left = Eq_State1(p_left, e_left, Type_of_state, 1.4, 1.0);
	rho_right = Eq_State1(p_right, e_right, Type_of_state, 1.4, 1.0);

	mas_particle = 1.0 / (Particle_on_length * Particle_on_length * Particle_on_length);
	dlh_left = pow(mas_particle/rho_left,1/3.0);
	dlh_right = pow(mas_particle / rho_right, 1 / 3.0);

	p = -1;

	Im = int((0.0 - X_min) / dlh_left) +1;
	Jm = int((Y_max - Y_min) / dlh_left) +1;
	Km = int((Z_max - Z_min) / dlh_left) +1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = 0.0 - (double)(i * dlh_left);
				y_temp = Y_min + (double)(j * dlh_left);
				z_temp = Z_min + (double)(k * dlh_left);
				if ((abs(x_temp) <= 1.0) && (abs(y_temp) <= 1.0) && (abs(z_temp) <=1.0))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_left;
					p_gas[p] = p_left;
					rho_gas[p] = rho_left;
					mas_gas[p] = mas_particle;
					}
				if ((abs(x_temp) <= 1.0 - border_length) && (abs(y_temp) <= 1.0 - border_length) && (abs(z_temp) <= 1.0 - border_length))
					{
						Ind_gas[p] = 0;
					}
				else
					{
						Ind_gas[p] = 1;
					}


				};

	Im = int((X_max - 0.0) / dlh_right) +1;
	Jm = int((Y_max - Y_min) / dlh_right) +1;
	Km = int((Z_max - Z_min) / dlh_right) +1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = dlh_right + (double)(i * dlh_right);
				y_temp = Y_min + (double)(j * dlh_right);
				z_temp = Z_min + (double)(k * dlh_right);
				if ((abs(x_temp) <= 1.0) && (abs(y_temp) <= 1.0) && (abs(z_temp) <= 1.0))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_right;
					p_gas[p] = p_right;
					rho_gas[p] = rho_right;
					mas_gas[p] = mas_particle;
				}
				if ((abs(x_temp) <= 1.0 - border_length) && (abs(y_temp) <= 1.0 - border_length) && (abs(z_temp) <= 1.0 - border_length))
				{
					Ind_gas[p] = 0;
				}
				else
				{
					Ind_gas[p] = 1;
				}

			};

	
	Pr = p;

	Pm = p;
}

void init_Sod_X_e()
{
	double x_temp, y_temp, z_temp;
	double p_left, p_right, e_left, e_right, rho_left, rho_right, dlh_left, dlh_right, mas_particle;

	double border_length = 2.5 * h;
	double X_cube = 1.0, Y_cube = 0.8, Z_cube = 0.8;

	e_left = 2.5;
	e_right = 2.0;
	rho_left = 1.0;  
	rho_right = 1.0;  
	p_left = Eq_State(rho_left, e_left, Type_of_state, 1.4, 1.0);
	p_right = Eq_State(rho_right, e_right, Type_of_state, 1.4, 1.0);


	mas_particle = 1.0 / (Particle_on_length * Particle_on_length * Particle_on_length);
	dlh_left = pow(mas_particle / rho_left, 1 / 3.0);
	dlh_right = pow(mas_particle / rho_right, 1 / 3.0);

	p = -1;

	Im = int((0.0 - X_min) / dlh_left) + 1;
	Jm = int((Y_max - Y_min) / dlh_left) + 1;
	Km = int((Z_max - Z_min) / dlh_left) + 1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = 0.0 - (double)(i * dlh_left);
				y_temp = Y_min + (double)(j * dlh_left);
				z_temp = Z_min + (double)(k * dlh_left);
				if ((abs(x_temp) <= X_cube) && (abs(y_temp) <= Y_cube) && (abs(z_temp) <= Z_cube))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_left;
					p_gas[p] = p_left;
					rho_gas[p] = rho_left;
					mas_gas[p] = mas_particle;
				}
				if ((abs(x_temp) <= X_cube - border_length) && (abs(y_temp) <= X_cube - border_length) && (abs(z_temp) <= Z_cube - border_length))
				{
					Ind_gas[p] = 0;
				}
				else
				{
					Ind_gas[p] = 1;
				}


			};

	Im = int((X_max - 0.0) / dlh_right) + 1;
	Jm = int((Y_max - Y_min) / dlh_right) + 1;
	Km = int((Z_max - Z_min) / dlh_right) + 1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = dlh_right + (double)(i * dlh_right);
				y_temp = Y_min + (double)(j * dlh_right);
				z_temp = Z_min + (double)(k * dlh_right);
				if ((abs(x_temp) <= X_cube) && (abs(y_temp) <= Y_cube) && (abs(z_temp) <= Z_cube))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_right;
					p_gas[p] = p_right;
					rho_gas[p] = rho_right;
					mas_gas[p] = mas_particle;
				}
				if ((abs(x_temp) <= X_cube - border_length) && (abs(y_temp) <= X_cube - border_length) && (abs(z_temp) <= Z_cube - border_length))
				{
					Ind_gas[p] = 0;
				}
				else
				{
					Ind_gas[p] = 1;
				}

			};


	Pr = p;

	Pm = p;
}


void init_Dust_Ball()
{
	double x_temp, y_temp, z_temp;
	int dust_id;

	p = -1;
	Im = int((X_max - X_min) * Particle_on_length);
	Jm = int((Y_max - Y_min) * Particle_on_length);
	Km = int((Z_max - Z_min) * Particle_on_length);

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = X_min + (double)(i * dlh);
				y_temp = Y_min + (double)(j * dlh);
				z_temp = Z_min + (double)(k * dlh);
				if (x_temp * x_temp + y_temp * y_temp + z_temp * z_temp <= 1.0) {
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					mas_gas[p] = 1.0 / (Particle_on_length * Particle_on_length * Particle_on_length);
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = 2.5;
					p_gas[p] = 1.0;
					rho_gas[p] = 1.0;
					Ind_gas[p] = 0;
				};
			}

	Pr = p;

	Pm = p;

	//Num_dust_sort = 2;
	Coeff_h_dust_cell = 0.5;
	mas_gas_dust_1 = 0.001;
	R_dust_1 = 0.001;
	mas_gas_dust_2 = 0.001;
	R_dust_2 = 0.001;


		x_dust_1 = new double[Pm + 1];
		y_dust_1 = new double[Pm + 1];
		z_dust_1 = new double[Pm + 1];
		h_dust_1 = new double[Pm + 1];
		mas_dust_1 = new double[Pm + 1];
		rho_dust_1 = new double[Pm + 1];
		Vx_dust_1 = new double[Pm + 1];
		Vy_dust_1 = new double[Pm + 1];
		Vz_dust_1 = new double[Pm + 1];
		Ax_dust_1 = new double[Pm + 1];
		Ay_dust_1 = new double[Pm + 1];
		Az_dust_1 = new double[Pm + 1];
		ind_dust_1 = new int[Pm + 1];
		t_stop_1 = new double[Pm + 1];
		Nn_dust_1 = new int[Pm + 1]; // Number of next particle in cell
		Cell_dust_1 = new int[Number_of_fihd_cells + 1]; // Number of first particle
		Ne_dust_1 = new int[Number_of_fihd_cells + 1]; //  Number of last particle
	
		x_dust_2 = new double[Pm + 1];
		y_dust_2 = new double[Pm + 1];
		z_dust_2 = new double[Pm + 1];
		h_dust_2 = new double[Pm + 1];
		mas_dust_2 = new double[Pm + 1];
		rho_dust_2 = new double[Pm + 1];
		Vx_dust_2 = new double[Pm + 1];
		Vy_dust_2 = new double[Pm + 1];
		Vz_dust_2 = new double[Pm + 1];
		Ax_dust_2 = new double[Pm + 1];
		Ay_dust_2 = new double[Pm + 1];
		Az_dust_2 = new double[Pm + 1];
		ind_dust_2 = new int[Pm + 1];
		t_stop_2 = new double[Pm + 1];
		Nn_dust_2 = new int[Pm + 1]; // Number of next particle in cell
		Cell_dust_2 = new int[Number_of_fihd_cells + 1]; // Number of first particle
		Ne_dust_2 = new int[Number_of_fihd_cells + 1]; //  Number of last particle
	

		for (p = 0; p <= Pm; p++)
		{
			
			x_dust_1[p] = x_gas[p];
			y_dust_1[p] = y_gas[p]; 
			z_dust_1[p] = z_gas[p];
			h_dust_1[p] = h;
			mas_dust_1[p] = mas_gas[p]* mas_gas_dust_1; 
			rho_dust_1[p] = rho_gas[p] * mas_gas_dust_1; 
			Vx_dust_1[p] = 0.0;
			Vy_dust_1[p] = 0.0;
			Vz_dust_1[p] = 0.0;
			Ax_dust_1[p] = 0.0;
			Ay_dust_1[p] = 0.0;
			Az_dust_1[p] = 0.0;
			ind_dust_1[p] = Ind_gas[p];
			t_stop_1[p] = Eq_t_stop(R_dust_1, p_gas[p], rho_gas[p], Gam_g);

			x_dust_2[p] = x_gas[p];
			y_dust_2[p] = y_gas[p];
			z_dust_2[p] = z_gas[p];
			h_dust_2[p] = h;
			mas_dust_2[p] = mas_gas[p] * mas_gas_dust_2;
			rho_dust_2[p] = rho_gas[p] * mas_gas_dust_2;
			Vx_dust_2[p] = 0.0;
			Vy_dust_2[p] = 0.0;
			Vz_dust_2[p] = 0.0;
			Ax_dust_2[p] = 0.0;
			Ay_dust_2[p] = 0.0;
			Az_dust_2[p] = 0.0;
			ind_dust_2[p] = Ind_gas[p];
			t_stop_2[p] = Eq_t_stop(R_dust_2, p_gas[p], rho_gas[p], Gam_g);

		}

	average_cell_width = Coeff_h_dust_cell * h; 
	Number_of_average_cell_x = int((X_max - X_min) / average_cell_width);
	Number_of_average_cell_y = int((Y_max - Y_min) / average_cell_width);
	Number_of_average_cell_z = int((Z_max - Z_min) / average_cell_width);

	Number_of_average_cell = (Number_of_average_cell_x + 1) * (Number_of_average_cell_y + 1) * (Number_of_average_cell_z + 1);
//	x_dust_cell = new double[Number_of_dust_cell + 1];
//	y_dust_cell = new double[Number_of_dust_cell + 1];
//	z_dust_cell = new double[Number_of_dust_cell + 1];
	
	Vx_g_average = new double[Number_of_average_cell + 1];
	Vy_g_average = new double[Number_of_average_cell + 1];
	Vz_g_average = new double[Number_of_average_cell + 1];
	rho_g_average = new double[Number_of_average_cell + 1];
	e_g_average = new double[Number_of_average_cell + 1];
	g_average_count = new int[Number_of_average_cell + 1];
	Psi_av_new_x = new double[Number_of_average_cell + 1];
	Psi_av_new_y = new double[Number_of_average_cell + 1];
	Psi_av_new_z = new double[Number_of_average_cell + 1];
	v_av_new = new double[Number_of_average_cell + 1];
	y_av_new_x = new double[Number_of_average_cell + 1];
	y_av_new_y = new double[Number_of_average_cell + 1];
	y_av_new_z = new double[Number_of_average_cell + 1];
	y_av_x = new double[Number_of_average_cell + 1];
	y_av_y = new double[Number_of_average_cell + 1];
	y_av_z = new double[Number_of_average_cell + 1];
	beta_cell = new double[Number_of_average_cell + 1];

	Vx_d_average_1 = new double[Number_of_average_cell + 1];
	Vy_d_average_1 = new double[Number_of_average_cell + 1];
	Vz_d_average_1 = new double[Number_of_average_cell + 1];
	rho_d_average_1 = new double[Number_of_average_cell + 1];
	eps_cell_1 = new double[Number_of_average_cell + 1];
	d_average_count_1 = new int[Number_of_average_cell + 1];
	t_stop_average_1 = new double[Number_of_average_cell + 1];
	x_av_new_x_1 = new double[Number_of_average_cell + 1];
	x_av_new_y_1 = new double[Number_of_average_cell + 1];
	x_av_new_z_1 = new double[Number_of_average_cell + 1];
	x_av_x_1 = new double[Number_of_average_cell + 1];
	x_av_y_1 = new double[Number_of_average_cell + 1];
	x_av_z_1 = new double[Number_of_average_cell + 1];
	u_av_new_1 = new double[Number_of_average_cell + 1];
	b_cell_1 = new double[Number_of_average_cell + 1];
	
	Vx_d_average_2 = new double[Number_of_average_cell + 1];
	Vy_d_average_2 = new double[Number_of_average_cell + 1];
	Vz_d_average_2 = new double[Number_of_average_cell + 1];
	rho_d_average_2 = new double[Number_of_average_cell + 1];
	eps_cell_2 = new double[Number_of_average_cell + 1];
	d_average_count_2 = new int[Number_of_average_cell + 1];
	t_stop_average_2 = new double[Number_of_average_cell + 1];
	x_av_new_x_2 = new double[Number_of_average_cell + 1];
	x_av_new_y_2 = new double[Number_of_average_cell + 1];
	x_av_new_z_2 = new double[Number_of_average_cell + 1];
	x_av_x_2 = new double[Number_of_average_cell + 1];
	x_av_y_2 = new double[Number_of_average_cell + 1];
	x_av_z_2 = new double[Number_of_average_cell + 1];
	u_av_new_2 = new double[Number_of_average_cell + 1];
	b_cell_2 = new double[Number_of_average_cell + 1];
		
	for (i = 0; i <= Number_of_average_cell; i++)
	{
		//x_dust_cell[i] = (double) X_min + i * cell_dust_width;
		//x_dust_cell[i] = (double) Y_min + i * cell_dust_width;
		//z_dust_cell[i] = (double) Y_min + i * cell_dust_width;
		Vx_g_average[i] = 0.0; 
		Vy_g_average[i] = 0.0; 
		Vz_g_average[i] = 0.0;
		rho_g_average[i] = 0.0;
		e_g_average[i] = 0.0;
		g_average_count[i] = 0;
		Psi_av_new_x[i] = 0.0;
		Psi_av_new_y[i] = 0.0;
		Psi_av_new_z[i] = 0.0;
		v_av_new[i] = 0.0;
		y_av_new_x[i] = 0.0;
		y_av_new_y[i] = 0.0;
		y_av_new_y[i] = 0.0;
		y_av_x[i] = 0.0;
		y_av_y[i] = 0.0;
		y_av_z[i] = 0.0;

		Vx_d_average_1[i] = 0.0; 
		Vy_d_average_1[i] = 0.0; 
		Vz_d_average_1[i] = 0.0;
		rho_d_average_1[i] = 0.0;
		eps_cell_1[i] = 0.0;
		t_stop_average_1[i] = 0.0;
		d_average_count_1[i] = 0;
		x_av_new_x_1[i] = 0.0;
		x_av_new_y_1[i] = 0.0;
		x_av_new_z_1[i] = 0.0;
		x_av_x_1[i] = 0.0;
		x_av_y_1[i] = 0.0;
		x_av_z_1[i] = 0.0;
		u_av_new_1[i] = 0.0;
		b_cell_1[i] = 0.0;

		Vx_d_average_2[i] = 0.0;
		Vy_d_average_2[i] = 0.0;
		Vz_d_average_2[i] = 0.0;
		rho_d_average_2[i] = 0.0;
		eps_cell_2[i] = 0.0;
		t_stop_average_2[i] = 0.0;
		d_average_count_2[i] = 0;
		x_av_new_x_2[i] = 0.0;
		x_av_new_y_2[i] = 0.0;
		x_av_new_z_2[i] = 0.0;
		x_av_x_2[i] = 0.0;
		x_av_y_2[i] = 0.0;
		x_av_z_2[i] = 0.0;
		u_av_new_2[i] = 0.0;
		b_cell_2[i] = 0.0;
		
	}

}


void init_Sod_X_Dust()
{
	double x_temp, y_temp, z_temp;
	double p_left, p_right, e_left, e_right, rho_left, rho_right, dlh_left, dlh_right, mas_particle;

	double border_length = 2.5 * h;
	double X_cube = 1.0, Y_cube = 0.8, Z_cube = 0.8;

	e_left = 2.5;
	e_right = 2.0;
	rho_left = 1.0;
	rho_right = 1.0;
	p_left = Eq_State(rho_left, e_left, Type_of_state, 1.4, 1.0);
	p_right = Eq_State(rho_right, e_right, Type_of_state, 1.4, 1.0);


	mas_particle = 1.0 / (Particle_on_length * Particle_on_length * Particle_on_length);
	dlh_left = pow(mas_particle / rho_left, 1 / 3.0);
	dlh_right = pow(mas_particle / rho_right, 1 / 3.0);

	p = -1;

	Im = int((0.0 - X_min) / dlh_left) + 1;
	Jm = int((Y_max - Y_min) / dlh_left) + 1;
	Km = int((Z_max - Z_min) / dlh_left) + 1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = 0.0 - (double)(i * dlh_left);
				y_temp = Y_min + (double)(j * dlh_left);
				z_temp = Z_min + (double)(k * dlh_left);
				if ((abs(x_temp) <= X_cube) && (abs(y_temp) <= Y_cube) && (abs(z_temp) <= Z_cube))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_left;
					p_gas[p] = p_left;
					rho_gas[p] = rho_left;
					mas_gas[p] = mas_particle;
				}
				if ((abs(x_temp) <= X_cube - border_length) && (abs(y_temp) <= X_cube - border_length) && (abs(z_temp) <= Z_cube - border_length))
				{
					Ind_gas[p] = 0;
				}
				else
				{
					Ind_gas[p] = 1;
				}


			};

	Im = int((X_max - 0.0) / dlh_right) + 1;
	Jm = int((Y_max - Y_min) / dlh_right) + 1;
	Km = int((Z_max - Z_min) / dlh_right) + 1;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = dlh_right + (double)(i * dlh_right);
				y_temp = Y_min + (double)(j * dlh_right);
				z_temp = Z_min + (double)(k * dlh_right);
				if ((abs(x_temp) <= X_cube) && (abs(y_temp) <= Y_cube) && (abs(z_temp) <= Z_cube))
				{
					p = p + 1;
					x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
					Vx_gas[p] = 0.0;
					Vy_gas[p] = 0.0;
					Vz_gas[p] = 0.0;
					Ax_gas[p] = 0.0;
					Ay_gas[p] = 0.0;
					Az_gas[p] = 0.0;
					e_gas[p] = e_right;
					p_gas[p] = p_right;
					rho_gas[p] = rho_right;
					mas_gas[p] = mas_particle;
				}
				if ((abs(x_temp) <= X_cube - border_length) && (abs(y_temp) <= X_cube - border_length) && (abs(z_temp) <= Z_cube - border_length))
				{
					Ind_gas[p] = 0;
				}
				else
				{
					Ind_gas[p] = 1;
				}

			};


	Pr = p;

	Pm = p;

	//Num_dust_sort = 2;
	Coeff_h_dust_cell = 0.5;
	mas_gas_dust_1 = 0.75;
	R_dust_1 = 0.01;
	mas_gas_dust_2 = 0.5;
	R_dust_2 = 1.0;


	x_dust_1 = new double[Pm + 1];
	y_dust_1 = new double[Pm + 1];
	z_dust_1 = new double[Pm + 1];
	h_dust_1 = new double[Pm + 1];
	mas_dust_1 = new double[Pm + 1];
	rho_dust_1 = new double[Pm + 1];
	Vx_dust_1 = new double[Pm + 1];
	Vy_dust_1 = new double[Pm + 1];
	Vz_dust_1 = new double[Pm + 1];
	Ax_dust_1 = new double[Pm + 1];
	Ay_dust_1 = new double[Pm + 1];
	Az_dust_1 = new double[Pm + 1];
	ind_dust_1 = new int[Pm + 1];
	t_stop_1 = new double[Pm + 1];
	Nn_dust_1 = new int[Pm + 1]; // Number of next particle in cell
	Cell_dust_1 = new int[Number_of_fihd_cells + 1]; // Number of first particle
	Ne_dust_1 = new int[Number_of_fihd_cells + 1]; //  Number of last particle

	x_dust_2 = new double[Pm + 1];
	y_dust_2 = new double[Pm + 1];
	z_dust_2 = new double[Pm + 1];
	h_dust_2 = new double[Pm + 1];
	mas_dust_2 = new double[Pm + 1];
	rho_dust_2 = new double[Pm + 1];
	Vx_dust_2 = new double[Pm + 1];
	Vy_dust_2 = new double[Pm + 1];
	Vz_dust_2 = new double[Pm + 1];
	Ax_dust_2 = new double[Pm + 1];
	Ay_dust_2 = new double[Pm + 1];
	Az_dust_2 = new double[Pm + 1];
	ind_dust_2 = new int[Pm + 1];
	t_stop_2 = new double[Pm + 1];
	Nn_dust_2 = new int[Pm + 1]; // Number of next particle in cell
	Cell_dust_2 = new int[Number_of_fihd_cells + 1]; // Number of first particle
	Ne_dust_2 = new int[Number_of_fihd_cells + 1]; //  Number of last particle


	for (p = 0; p <= Pm; p++)
	{

		x_dust_1[p] = x_gas[p];
		y_dust_1[p] = y_gas[p];
		z_dust_1[p] = z_gas[p];
		h_dust_1[p] = h;
		mas_dust_1[p] = mas_gas[p] * mas_gas_dust_1;
		rho_dust_1[p] = rho_gas[p] * mas_gas_dust_1;
		Vx_dust_1[p] = 0.0;
		Vy_dust_1[p] = 0.0;
		Vz_dust_1[p] = 0.0;
		Ax_dust_1[p] = 0.0;
		Ay_dust_1[p] = 0.0;
		Az_dust_1[p] = 0.0;
		ind_dust_1[p] = Ind_gas[p];
		t_stop_1[p] = Eq_t_stop(R_dust_1, p_gas[p], rho_gas[p], Gam_g);

		x_dust_2[p] = x_gas[p];
		y_dust_2[p] = y_gas[p];
		z_dust_2[p] = z_gas[p];
		h_dust_2[p] = h;
		mas_dust_2[p] = mas_gas[p] * mas_gas_dust_2;
		rho_dust_2[p] = rho_gas[p] * mas_gas_dust_2;
		Vx_dust_2[p] = 0.0;
		Vy_dust_2[p] = 0.0;
		Vz_dust_2[p] = 0.0;
		Ax_dust_2[p] = 0.0;
		Ay_dust_2[p] = 0.0;
		Az_dust_2[p] = 0.0;
		ind_dust_2[p] = Ind_gas[p];
		t_stop_2[p] = Eq_t_stop(R_dust_2, p_gas[p], rho_gas[p], Gam_g);

	}

	average_cell_width = Coeff_h_dust_cell * h;
	Number_of_average_cell_x = int((X_max - X_min) / average_cell_width);
	Number_of_average_cell_y = int((Y_max - Y_min) / average_cell_width);
	Number_of_average_cell_z = int((Z_max - Z_min) / average_cell_width);

	Number_of_average_cell = (Number_of_average_cell_x + 1) * (Number_of_average_cell_y + 1) * (Number_of_average_cell_z + 1);
	//	x_dust_cell = new double[Number_of_dust_cell + 1];
	//	y_dust_cell = new double[Number_of_dust_cell + 1];
	//	z_dust_cell = new double[Number_of_dust_cell + 1];

	Vx_g_average = new double[Number_of_average_cell + 1];
	Vy_g_average = new double[Number_of_average_cell + 1];
	Vz_g_average = new double[Number_of_average_cell + 1];
	rho_g_average = new double[Number_of_average_cell + 1];
	e_g_average = new double[Number_of_average_cell + 1];
	g_average_count = new int[Number_of_average_cell + 1];
	Psi_av_new_x = new double[Number_of_average_cell + 1];
	Psi_av_new_y = new double[Number_of_average_cell + 1];
	Psi_av_new_z = new double[Number_of_average_cell + 1];
	v_av_new = new double[Number_of_average_cell + 1];
	y_av_new_x = new double[Number_of_average_cell + 1];
	y_av_new_y = new double[Number_of_average_cell + 1];
	y_av_new_z = new double[Number_of_average_cell + 1];
	y_av_x = new double[Number_of_average_cell + 1];
	y_av_y = new double[Number_of_average_cell + 1];
	y_av_z = new double[Number_of_average_cell + 1];
	beta_cell = new double[Number_of_average_cell + 1];

	Vx_d_average_1 = new double[Number_of_average_cell + 1];
	Vy_d_average_1 = new double[Number_of_average_cell + 1];
	Vz_d_average_1 = new double[Number_of_average_cell + 1];
	rho_d_average_1 = new double[Number_of_average_cell + 1];
	eps_cell_1 = new double[Number_of_average_cell + 1];
	d_average_count_1 = new int[Number_of_average_cell + 1];
	t_stop_average_1 = new double[Number_of_average_cell + 1];
	x_av_new_x_1 = new double[Number_of_average_cell + 1];
	x_av_new_y_1 = new double[Number_of_average_cell + 1];
	x_av_new_z_1 = new double[Number_of_average_cell + 1];
	x_av_x_1 = new double[Number_of_average_cell + 1];
	x_av_y_1 = new double[Number_of_average_cell + 1];
	x_av_z_1 = new double[Number_of_average_cell + 1];
	u_av_new_1 = new double[Number_of_average_cell + 1];
	b_cell_1 = new double[Number_of_average_cell + 1];

	Vx_d_average_2 = new double[Number_of_average_cell + 1];
	Vy_d_average_2 = new double[Number_of_average_cell + 1];
	Vz_d_average_2 = new double[Number_of_average_cell + 1];
	rho_d_average_2 = new double[Number_of_average_cell + 1];
	eps_cell_2 = new double[Number_of_average_cell + 1];
	d_average_count_2 = new int[Number_of_average_cell + 1];
	t_stop_average_2 = new double[Number_of_average_cell + 1];
	x_av_new_x_2 = new double[Number_of_average_cell + 1];
	x_av_new_y_2 = new double[Number_of_average_cell + 1];
	x_av_new_z_2 = new double[Number_of_average_cell + 1];
	x_av_x_2 = new double[Number_of_average_cell + 1];
	x_av_y_2 = new double[Number_of_average_cell + 1];
	x_av_z_2 = new double[Number_of_average_cell + 1];
	u_av_new_2 = new double[Number_of_average_cell + 1];
	b_cell_2 = new double[Number_of_average_cell + 1];

	for (i = 0; i <= Number_of_average_cell; i++)
	{
		//x_dust_cell[i] = (double) X_min + i * cell_dust_width;
		//x_dust_cell[i] = (double) Y_min + i * cell_dust_width;
		//z_dust_cell[i] = (double) Y_min + i * cell_dust_width;
		Vx_g_average[i] = 0.0;
		Vy_g_average[i] = 0.0;
		Vz_g_average[i] = 0.0;
		rho_g_average[i] = 0.0;
		e_g_average[i] = 0.0;
		g_average_count[i] = 0;
		Psi_av_new_x[i] = 0.0;
		Psi_av_new_y[i] = 0.0;
		Psi_av_new_z[i] = 0.0;
		v_av_new[i] = 0.0;
		y_av_new_x[i] = 0.0;
		y_av_new_y[i] = 0.0;
		y_av_new_y[i] = 0.0;
		y_av_x[i] = 0.0;
		y_av_y[i] = 0.0;
		y_av_z[i] = 0.0;

		Vx_d_average_1[i] = 0.0;
		Vy_d_average_1[i] = 0.0;
		Vz_d_average_1[i] = 0.0;
		rho_d_average_1[i] = 0.0;
		eps_cell_1[i] = 0.0;
		t_stop_average_1[i] = 0.0;
		d_average_count_1[i] = 0;
		x_av_new_x_1[i] = 0.0;
		x_av_new_y_1[i] = 0.0;
		x_av_new_z_1[i] = 0.0;
		x_av_x_1[i] = 0.0;
		x_av_y_1[i] = 0.0;
		x_av_z_1[i] = 0.0;
		u_av_new_1[i] = 0.0;
		b_cell_1[i] = 0.0;

		Vx_d_average_2[i] = 0.0;
		Vy_d_average_2[i] = 0.0;
		Vz_d_average_2[i] = 0.0;
		rho_d_average_2[i] = 0.0;
		eps_cell_2[i] = 0.0;
		t_stop_average_2[i] = 0.0;
		d_average_count_2[i] = 0;
		x_av_new_x_2[i] = 0.0;
		x_av_new_y_2[i] = 0.0;
		x_av_new_z_2[i] = 0.0;
		x_av_x_2[i] = 0.0;
		x_av_y_2[i] = 0.0;
		x_av_z_2[i] = 0.0;
		u_av_new_2[i] = 0.0;
		b_cell_2[i] = 0.0;

	}

}


int main()

{

	double tmp1;

	FILE *ini_file, * out_file;
	char s[128];
//	fopen_s(&ini_file, "Init.txt", "r");
	ini_file = fopen("Init.txt", "r");
	fgets(s, 128, ini_file);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &X_min);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &X_max);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Y_min);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Y_max);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Z_min);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Z_max);
	fgets(s, 128, ini_file); sscanf(s, "%d", &Type_of_state);
	fgets(s, 128, ini_file); sscanf(s, "%d", &Maximum_particle);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Particle_on_length);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &h);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Coeff_h_dust_cell);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &tau);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &T_end);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &T_out);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &alpha);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &beta);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &eps);


/*	fgets(s, 128, ini_file);	sscanf(s, "%lf", &Zm);
	fgets(s, 128, ini_file);	sscanf(s, "%lf", &c0);
	fgets(s, 128, ini_file);	sscanf(s, "%lf", &n);
	
	fgets(s, 128, ini_file); sscanf(s, "%lf", &p_in);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &p_out);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &tau);
	fgets(s, 128, ini_file);	sscanf(s, "%d", &Te);
	fgets(s, 128, ini_file);	sscanf(s, "%d", &File_int);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Geps);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &alpha);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &beta);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &eps);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &sigm);	
*/

	fclose(ini_file);

	Speed_of_sound_gas = 1.0;
	dlh = (double) 1.0 / Particle_on_length;
	Tm = 0.0;
	out_num = 0;

	// Memory

	x_gas = new double[Maximum_particle];
	y_gas = new double[Maximum_particle];
	z_gas = new double[Maximum_particle];
	p_gas = new double[Maximum_particle];
	rho_gas = new double[Maximum_particle];
	mas_gas = new double[Maximum_particle];
	Vx_gas = new double[Maximum_particle];
	Vy_gas = new double[Maximum_particle];
	Vz_gas = new double[Maximum_particle];
	Ax_gas = new double[Maximum_particle];
	Ay_gas = new double[Maximum_particle];
	Az_gas = new double[Maximum_particle];
	e_gas = new double[Maximum_particle];
	Ind_gas = new int[Maximum_particle];
	Nn_gas = new int[Maximum_particle]; // Number of next particle in cell

	Clx = X_min;
	Cly = Y_min;
	Clz = Z_min;
	Cmx = X_max;
	Cmy = Y_max;
	Cmz = Z_max;
	Clh = 2.0 * h; // Cell length 
	Cnx = int((Cmx - Clx) / Clh);
	Cny = int((Cmy - Cly) / Clh);
	Cnz = int((Cmz - Clz) / Clh);

	Number_of_fihd_cells = (Cnx + 1) * (Cny + 1) * (Cnz + 1);
	Cell_gas = new int[Number_of_fihd_cells + 1]; // Number of first particle
	Ne_gas = new int[Number_of_fihd_cells + 1]; //  Number of last particle



	// Particle placing

	// init_Ball(); 
	//  init_Sod_X();
	// init_Sod_X_mas();
	// init_Sod_X_e();
	// init_Dust_Ball();
	init_Sod_X_Dust();

	cudaSetDevice(1);

	cudaMalloc((void**)&dev_x_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_y_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_z_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Vx_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Vy_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Vz_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Ax_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Ay_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Az_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_rho_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_p_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_e_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_mas_gas, (Pm + 1) * sizeof(double));
	cudaMalloc((void**)&dev_ind_gas, (Pm + 1) * sizeof(int));
	cudaMalloc((void**)&dev_Cell_gas, (Number_of_fihd_cells+1) * sizeof(int));
	cudaMalloc((void**)&dev_Nn_gas, (Pm + 1) * sizeof(int));
	cudaMalloc((void**)&dev_e_temp, (Pm + 1) * sizeof(double));
	

		cudaMalloc((void**)&dev_x_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_y_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_z_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vx_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vy_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vz_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Ax_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Ay_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Az_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_rho_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_mas_dust_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_t_stop_1, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_ind_dust_1, (Pm + 1) * sizeof(int));
		cudaMalloc((void**)&dev_Cell_dust_1, (Number_of_fihd_cells+1) * sizeof(int));
		cudaMalloc((void**)&dev_Nn_dust_1, (Pm + 1) * sizeof(int));


		cudaMalloc((void**)&dev_x_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_y_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_z_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vx_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vy_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vz_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Ax_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Ay_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Az_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_rho_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_mas_dust_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_t_stop_2, (Pm + 1) * sizeof(double));
		cudaMalloc((void**)&dev_ind_dust_2, (Pm + 1) * sizeof(int));
		cudaMalloc((void**)&dev_Cell_dust_2, (Number_of_fihd_cells+1) * sizeof(int));
		cudaMalloc((void**)&dev_Nn_dust_2, (Pm + 1) * sizeof(int));
		
		cudaMalloc((void**)&dev_Psi_av_new_x, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Psi_av_new_y, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Psi_av_new_z, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_e_temp, (Pm + 1) * sizeof(double));

		cudaMalloc((void**)&dev_Vx_g_average, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vy_g_average, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vz_g_average, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vx_d_average_1, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vy_d_average_1, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vz_d_average_1, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vx_d_average_2, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vy_d_average_2, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_Vz_d_average_2, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_t_stop_average_1, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_t_stop_average_2, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_eps_cell_1, (Number_of_average_cell + 1) * sizeof(double));
		cudaMalloc((void**)&dev_eps_cell_2, (Number_of_average_cell + 1) * sizeof(double));


	dim3 gridSize = dim3(Pm / 256 + 1, 1, 1);  
	dim3 blockSize = dim3(16, 16, 1);

	Tm = 0.0;
	out_num = 0;

	Data_out(out_num);
	Data_out_dust(out_num);
	out_num = out_num + 1;


	cudaMemcpy(dev_x_gas, x_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y_gas, y_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_z_gas, z_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rho_gas, rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_p_gas, p_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mas_gas, mas_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_e_gas, e_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Vx_gas, Vx_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Vy_gas, Vy_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Vz_gas, Vz_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Ax_gas, Ax_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Ay_gas, Ay_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Az_gas, Az_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ind_gas, Ind_gas, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Cell_gas, Cell_gas, (Number_of_fihd_cells+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Nn_gas, Nn_gas, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);


	cudaMemcpy(dev_x_dust_1, x_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y_dust_1, y_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_z_dust_1, z_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Vx_dust_1, Vx_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Vy_dust_1, Vy_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Vz_dust_1, Vz_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Ax_dust_1, Ax_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Ay_dust_1, Ay_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Az_dust_1, Az_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rho_dust_1, rho_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t_stop_1, t_stop_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mas_dust_1, mas_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ind_dust_1, ind_dust_1, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Cell_dust_1, Cell_dust_1, (Number_of_fihd_cells+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Nn_dust_1, Nn_dust_1, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);

	cudaMemcpy(dev_x_dust_2, x_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y_dust_2, y_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_z_dust_2, z_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Vx_dust_2, Vx_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Vy_dust_2, Vy_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Vz_dust_2, Vz_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Ax_dust_2, Ax_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Ay_dust_2, Ay_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Az_dust_2, Az_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_rho_dust_2, rho_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_t_stop_2, t_stop_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_mas_dust_2, mas_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_ind_dust_2, ind_dust_2, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Cell_dust_2, Cell_dust_2, (Number_of_fihd_cells+1) * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_Nn_dust_2, Nn_dust_2, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);


	do
	{
		Tm = Tm + tau;
		printf("Time %5.3lf \n", Tm);


		// Allocation particled to cells

		for (i = 0; i <= Number_of_fihd_cells; i++)
		{
			Cell_gas[i] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			Nn_gas[p] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			i = int((x_gas[p] - Clx) / Clh) + (int((y_gas[p] - Cly) / Clh)) * Cnx + (int((z_gas[p] - Clz) / Clh)) * Cnx * Cny;
			if ((i >= 0) && (i <= Number_of_fihd_cells))
			{
				if (Cell_gas[i] == -1) { Cell_gas[i] = p; Ne_gas[i] = p; Nn_gas[p] = -1; }
				else { Nn_gas[Ne_gas[i]] = p; Ne_gas[i] = p; Nn_gas[p] = -1; };
			}
		}

		for (i = 0; i <= Number_of_fihd_cells; i++)
		{
			Cell_dust_1[i] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			Nn_dust_1[p] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			i = int((x_dust_1[p] - Clx) / Clh) + (int((y_dust_1[p] - Cly) / Clh)) * Cnx + (int((z_dust_1[p] - Clz) / Clh)) * Cnx * Cny;
			if ((i >= 0) && (i <= Number_of_fihd_cells))
			{
				if (Cell_dust_1[i] == -1) { Cell_dust_1[i] = p; Ne_dust_1[i] = p; Nn_dust_1[p] = -1; }
				else { Nn_dust_1[Ne_dust_1[i]] = p; Ne_dust_1[i] = p; Nn_dust_1[p] = -1; };
			}
		}

		for (i = 0; i <= Number_of_fihd_cells; i++)
		{
			Cell_dust_2[i] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			Nn_dust_2[p] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			i = int((x_dust_2[p] - Clx) / Clh) + (int((y_dust_2[p] - Cly) / Clh)) * Cnx + (int((z_dust_2[p] - Clz) / Clh)) * Cnx * Cny;
			if ((i >= 0) && (i <= Number_of_fihd_cells))
			{
				if (Cell_dust_2[i] == -1) { Cell_dust_2[i] = p; Ne_dust_2[i] = p; Nn_dust_2[p] = -1; }
				else { Nn_dust_2[Ne_dust_2[i]] = p; Ne_dust_2[i] = p; Nn_dust_2[p] = -1; };
			}
		}





		cudaMemcpy(dev_Cell_gas, Cell_gas, (Number_of_fihd_cells+1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Nn_gas, Nn_gas, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_Cell_dust_1, Cell_dust_1, (Number_of_fihd_cells+1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Nn_dust_1, Nn_dust_1, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_Cell_dust_2, Cell_dust_2, (Number_of_fihd_cells+1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Nn_dust_2, Nn_dust_2, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);


		Rho_InitKernel_IDIC << <gridSize, blockSize >> > (dev_rho_gas, dev_rho_dust_1, dev_rho_dust_2, dev_ind_gas, Pm);
		cudaDeviceSynchronize();

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{

					Rho_Kernel_IDIC << <gridSize, blockSize >> > (dev_x_gas, dev_y_gas, dev_z_gas, dev_rho_gas, dev_mas_gas, dev_ind_gas, dev_Cell_gas, dev_Nn_gas, Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Number_of_fihd_cells, h, k, l, g);
					cudaDeviceSynchronize();
				}

		PKernel << <gridSize, blockSize >> > (dev_rho_gas, dev_p_gas, dev_e_gas, dev_ind_gas, Pm, Type_of_state, Gam_g, Speed_of_sound_gas);
		cudaDeviceSynchronize();

		cudaMemcpy(rho_gas, dev_rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(p_gas, dev_p_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{
					Rho_Kernel_IDIC << <gridSize, blockSize >> > (dev_x_dust_1, dev_y_dust_1, dev_z_dust_1, dev_rho_dust_1, dev_mas_dust_1, dev_ind_dust_1, dev_Cell_dust_1, dev_Nn_dust_1, Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Number_of_fihd_cells, h, k, l, g);
					cudaDeviceSynchronize();
				}

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{
					Rho_Kernel_IDIC << <gridSize, blockSize >> > (dev_x_dust_2, dev_y_dust_2, dev_z_dust_2, dev_rho_dust_2, dev_mas_dust_2, dev_ind_dust_2, dev_Cell_dust_2, dev_Nn_dust_2, Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Number_of_fihd_cells, h, k, l, g);
					cudaDeviceSynchronize();
				}

		cudaMemcpy(rho_dust_1, dev_rho_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(rho_dust_2, dev_rho_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);

		
		t_stop_Kernel_IDIC << <gridSize, blockSize >> > (dev_t_stop_1, dev_p_gas, dev_rho_gas, dev_ind_dust_1, Pm, R_dust_1, Gam_g);
		cudaDeviceSynchronize();
				
		t_stop_Kernel_IDIC << <gridSize, blockSize >> > (dev_t_stop_2, dev_p_gas, dev_rho_gas, dev_ind_dust_2, Pm, R_dust_2, Gam_g);
		 cudaDeviceSynchronize();

		cudaMemcpy(t_stop_1, dev_t_stop_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(t_stop_2, dev_t_stop_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		
		

		for (i = 0; i <= Number_of_average_cell; i++)
		{
			Vx_g_average[i] = 0.0;
			Vy_g_average[i] = 0.0;
			Vz_g_average[i] = 0.0;
			rho_g_average[i] = 0.0;
			e_g_average[i] = 0.0;
			g_average_count[i] = 0;
			Psi_av_new_x[i] = 0.0;
			Psi_av_new_y[i] = 0.0;
			Psi_av_new_z[i] = 0.0;
			v_av_new[i] = 0.0;
			y_av_new_x[i] = 0.0;
			y_av_new_y[i] = 0.0;
			y_av_new_z[i] = 0.0;
			y_av_x[i] = 0.0;
			y_av_y[i] = 0.0;
			y_av_z[i] = 0.0;

			Vx_d_average_1[i] = 0.0;
			Vy_d_average_1[i] = 0.0;
			Vz_d_average_1[i] = 0.0;
			rho_d_average_1[i] = 0.0;
			eps_cell_1[i] = 0.0;
			t_stop_average_1[i] = 0.0;
			d_average_count_1[i] = 0;
			x_av_new_x_1[i] = 0.0;
			x_av_new_y_1[i] = 0.0;
			x_av_new_z_1[i] = 0.0;
			x_av_x_1[i] = 0.0;
			x_av_y_1[i] = 0.0;
			x_av_z_1[i] = 0.0;
			u_av_new_1[i] = 0.0;
			b_cell_1[i] = 0.0;

			Vx_d_average_2[i] = 0.0;
			Vy_d_average_2[i] = 0.0;
			Vz_d_average_2[i] = 0.0;
			rho_d_average_2[i] = 0.0;
			eps_cell_2[i] = 0.0;
			t_stop_average_2[i] = 0.0;
			d_average_count_2[i] = 0;
			x_av_new_x_2[i] = 0.0;
			x_av_new_y_2[i] = 0.0;
			x_av_new_z_2[i] = 0.0;
			x_av_x_2[i] = 0.0;
			x_av_y_2[i] = 0.0;
			x_av_z_2[i] = 0.0;
			u_av_new_2[i] = 0.0;
			b_cell_2[i] = 0.0;

		}


		for (p = 0; p <= Pm; p++)
		{
			cell_num = int((x_gas[p] - X_min) / average_cell_width) + (int((y_gas[p] - Y_min) / average_cell_width)) * Number_of_average_cell_x + (int((z_gas[p] - Z_min) / average_cell_width)) * Number_of_average_cell_x * Number_of_average_cell_y;
			g_average_count[cell_num] = g_average_count[cell_num] + 1;
			Vx_g_average[cell_num] = Vx_g_average[cell_num] + Vx_gas[p];
			Vy_g_average[cell_num] = Vy_g_average[cell_num] + Vy_gas[p];
			Vz_g_average[cell_num] = Vz_g_average[cell_num] + Vz_gas[p];
			rho_g_average[cell_num] = rho_g_average[cell_num] + rho_gas[p];
			e_g_average[cell_num] = e_g_average[cell_num] + e_gas[p];
		}


		for (i = 0; i <= Number_of_average_cell; i++)
		{
			if (g_average_count[i] > 0)
			{
				Vx_g_average[i] = Vx_g_average[i] / g_average_count[i];
				Vy_g_average[i] = Vy_g_average[i] / g_average_count[i];
				Vz_g_average[i] = Vz_g_average[i] / g_average_count[i];
				e_g_average[i] = e_g_average[i] / g_average_count[i];
			}
		}

		for (p = 0; p <= Pm; p++)
		{
			cell_num = int((x_dust_1[p] - X_min) / average_cell_width) + (int((y_dust_1[p] - Y_min) / average_cell_width)) * Number_of_average_cell_x + (int((z_dust_1[p] - Z_min) / average_cell_width)) * Number_of_average_cell_x * Number_of_average_cell_y;
			d_average_count_1[cell_num] = d_average_count_1[cell_num] + 1;
			t_stop_average_1[cell_num] = t_stop_average_1[cell_num] + t_stop_1[p];
			Vx_d_average_1[cell_num] = Vx_d_average_1[cell_num] + Vx_dust_1[p];
			Vy_d_average_1[cell_num] = Vy_d_average_1[cell_num] + Vy_dust_1[p];
			Vz_d_average_1[cell_num] = Vz_d_average_1[cell_num] + Vz_dust_1[p];
			rho_d_average_1[cell_num] = rho_d_average_1[cell_num] + rho_dust_1[p];
		}

		for (i = 0; i <= Number_of_average_cell; i++)
		{
			if (d_average_count_1[i] > 0)
			{
				Vx_d_average_1[i] = Vx_d_average_1[i] / d_average_count_1[i];
				Vy_d_average_1[i] = Vy_d_average_1[i] / d_average_count_1[i];
				Vz_d_average_1[i] = Vz_d_average_1[i] / d_average_count_1[i];
				rho_d_average_1[i] = rho_d_average_1[i] / d_average_count_1[i];
				t_stop_average_1[i] = t_stop_average_1[i] / d_average_count_1[i];
			}
			else
			{
				t_stop_average_1[i] = 1.0;
			}
		}

		
		for (p = 0; p <= Pm; p++)
		{
			cell_num = int((x_dust_2[p] - X_min) / average_cell_width) + (int((y_dust_2[p] - Y_min) / average_cell_width)) * Number_of_average_cell_x + (int((z_dust_2[p] - Z_min) / average_cell_width)) * Number_of_average_cell_x * Number_of_average_cell_y;
			d_average_count_2[cell_num] = d_average_count_2[cell_num] + 1;
			t_stop_average_2[cell_num] = t_stop_average_2[cell_num] + t_stop_2[p];
			Vx_d_average_2[cell_num] = Vx_d_average_2[cell_num] + Vx_dust_2[p];
			Vy_d_average_2[cell_num] = Vy_d_average_2[cell_num] + Vy_dust_2[p];
			Vz_d_average_2[cell_num] = Vz_d_average_2[cell_num] + Vz_dust_2[p];
			rho_d_average_2[cell_num] = rho_d_average_2[cell_num] + rho_dust_2[p];
		}

		for (i = 0; i <= Number_of_average_cell; i++)
		{
			if (d_average_count_2[i] > 0)
			{
				Vx_d_average_2[i] = Vx_d_average_2[i] / d_average_count_2[i];
				Vy_d_average_2[i] = Vy_d_average_2[i] / d_average_count_2[i];
				Vz_d_average_2[i] = Vz_d_average_2[i] / d_average_count_2[i];
				rho_d_average_2[i] = rho_d_average_2[i] / d_average_count_2[i];
				t_stop_average_2[i] = t_stop_average_2[i] / d_average_count_2[i];
			}
			else
			{
				t_stop_average_2[i] = 1.0;
			}
		}

		for (i = 0; i <= Number_of_average_cell; i++)
		{
			if ((g_average_count[i] > 0) && (d_average_count_1[i] > 0))
			{
				eps_cell_1[i] = d_average_count_1[i] * mas_gas_dust_1 / g_average_count[i];
			}
			else
			{
				eps_cell_1[i] = 0.5;
			}

			if ((g_average_count[i] > 0) && (d_average_count_1[i] > 0))
			{
				eps_cell_2[i] = d_average_count_2[i] * mas_gas_dust_2 / g_average_count[i];
			}
			else
			{
				eps_cell_2[i] = 0.33;
			}
		}
/*
		out_file = fopen("Data/Cell_av.dat", "wt");
		fprintf(out_file, "t=%5.3f \n", Tm);

		for (i = 0; i <= Number_of_average_cell; i++)
		{
			fprintf(out_file, "%d \t %d \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf  \n",
				i, g_average_count[i], Vx_g_average[i], Vy_g_average[i], Vz_g_average[i], e_g_average[i], d_average_count_1[i], Vx_d_average_1[i], Vy_d_average_1[i], Vz_d_average_1[i], rho_d_average_1[i], eps_cell_1[i], t_stop_average_1[i],
				d_average_count_2[i], Vx_d_average_2[i], Vy_d_average_2[i], Vz_d_average_2[i], rho_d_average_2[i], eps_cell_2[i], t_stop_average_2[i]);

		}
		fclose(out_file);

	*/	

	//	cudaMemcpy(dev_Psi_av_new_x, Psi_av_new_x, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
	//	cudaMemcpy(dev_Psi_av_new_y, Psi_av_new_y, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
	//	cudaMemcpy(dev_Psi_av_new_z, Psi_av_new_z, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);

	

		EnergyInitKernel_IDIC << <gridSize, blockSize >> > (dev_e_temp, dev_ind_gas, Pm);
		cudaDeviceSynchronize();

/*
		PsiInitKernel_IDIC << <gridSize, blockSize >> > (dev_Psi_av_new_x, dev_Psi_av_new_y, dev_Psi_av_new_z, Number_of_average_cell);
		cudaDeviceSynchronize();


		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{
					PsiKernel_IDIC << <gridSize, blockSize >> > (dev_x_gas, dev_y_gas, dev_z_gas, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_rho_gas, dev_p_gas, dev_mas_gas, dev_ind_gas, dev_Cell_gas, dev_Nn_gas, dev_e_temp, dev_Psi_av_new_x, dev_Psi_av_new_y, dev_Psi_av_new_z, Pm, Clx, Cly, Clz, Clh, Cnx, 
						Cny, Number_of_fihd_cells, h, X_min, Y_min, Z_min, average_cell_width, Number_of_average_cell_x, Number_of_average_cell_y, Number_of_average_cell, Gam_g, alpha, beta, eps, k, l, g);
					cudaDeviceSynchronize();
				}

		cudaMemcpy(Psi_av_new_x, dev_Psi_av_new_x, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Psi_av_new_y, dev_Psi_av_new_y, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Psi_av_new_z, dev_Psi_av_new_z, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyDeviceToHost);

*/

		Force_InitKernel_IDIC << <gridSize, blockSize >> > (dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_ind_gas, Pm);
		cudaDeviceSynchronize();

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{
					ForceKernel_IDIC << <gridSize, blockSize >> > (dev_x_gas, dev_y_gas, dev_z_gas, dev_rho_gas, dev_p_gas, dev_mas_gas, dev_e_temp, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_ind_gas, dev_Cell_gas, dev_Nn_gas, Pm, Clx, Cly, Clz, Clh, Cnx,
						Cny, Number_of_fihd_cells, h, tau , Gam_g, alpha, beta, eps, k, l, g);
					cudaDeviceSynchronize();
				}

		cudaMemcpy(Ax_gas, dev_Ax_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ay_gas, dev_Ay_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Az_gas, dev_Az_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
/*
		out_file = fopen("Data/A.dat", "wt");
		fprintf(out_file, "t=%5.3f \n", Tm);
		for (i = 0; i <= Pm; i++)
		{
			fprintf(out_file, "%d \t %10.8lf \t %10.8lf \t %10.8lf \n",
				i, Ax_gas[i], Ay_gas[i], Az_gas[i]);

		}
		fclose(out_file);
		
			out_file = fopen("Data/Cell_Psi.dat", "wt");
			fprintf(out_file, "t=%5.3f \n", Tm);
			for (i = 0; i <= Number_of_average_cell; i++)
			{
				fprintf(out_file, "%d \t %10.8lf \t %10.8lf \t %10.8lf \n",
					i, Psi_av_new_x[i], Psi_av_new_y[i], Psi_av_new_z[i]);

			}
			fclose(out_file);
*/
		

		for (p = 0; p <= Pm; p++)
		{
			cell_num = int((x_gas[p] - X_min) / average_cell_width) + (int((y_gas[p] - Y_min) / average_cell_width)) * Number_of_average_cell_x + (int((z_gas[p] - Z_min) / average_cell_width)) * Number_of_average_cell_x * Number_of_average_cell_y;
			Psi_av_new_x[cell_num] = Psi_av_new_x[cell_num] + Ax_gas[p];
			Psi_av_new_y[cell_num] = Psi_av_new_y[cell_num] + Ay_gas[p];
			Psi_av_new_z[cell_num] = Psi_av_new_z[cell_num] + Az_gas[p];
		}

		for (i = 0; i <= Number_of_average_cell; i++)
		{
			if (g_average_count[i] > 0)
			{
				Psi_av_new_x[i] = Psi_av_new_x[i] / g_average_count[i];
				Psi_av_new_y[i] = Psi_av_new_y[i] / g_average_count[i];
				Psi_av_new_z[i] = Psi_av_new_z[i] / g_average_count[i];
			}
			y_av_x[i] = Vx_g_average[i];
			y_av_y[i] = Vy_g_average[i];
			y_av_z[i] = Vz_g_average[i];
			
			x_av_x_1[i] = Vx_g_average[i] - Vx_d_average_1[i];
			x_av_y_1[i] = Vy_g_average[i] - Vy_d_average_1[i];
			x_av_z_1[i] = Vz_g_average[i] - Vz_d_average_1[i];
			
			x_av_x_2[i] = Vx_g_average[i] - Vx_d_average_2[i];
			x_av_y_2[i] = Vy_g_average[i] - Vy_d_average_2[i];
			x_av_z_2[i] = Vz_g_average[i] - Vz_d_average_2[i];

			y_av_x[i] = y_av_x[i] + eps_cell_1[i] * Vx_d_average_1[i] + eps_cell_2[i] * Vx_d_average_2[i];
			y_av_y[i] = y_av_y[i] + eps_cell_1[i] * Vy_d_average_1[i] + eps_cell_2[i] * Vy_d_average_2[i];
			y_av_z[i] = y_av_z[i] + eps_cell_1[i] * Vz_d_average_1[i] + eps_cell_2[i] * Vz_d_average_2[i];

			b_cell_1[i] = (t_stop_average_1[i] + tau) / (eps_cell_1[i] * tau);
			b_cell_2[i] = (t_stop_average_2[i] + tau) / (eps_cell_2[i] * tau);

		}
/*
		out_file = fopen("Data/Cell_Psi.dat", "wt");
		fprintf(out_file, "t=%5.3f \n", Tm);
		for (i = 0; i <= Number_of_average_cell; i++)
		{
			fprintf(out_file, "%d \t %10.8lf \t %10.8lf \t %10.8lf \n",
				i, Psi_av_new_x[i], Psi_av_new_y[i], Psi_av_new_z[i]);

		}
		fclose(out_file);
*/

		for (i = 0; i <= Number_of_average_cell; i++)
		{
			beta_cell[i] = 1.0;
			beta_cell[i] = beta_cell[i] + 1.0 / b_cell_1[i] + 1.0 / b_cell_2[i];
		}

		for (i = 0; i <= Number_of_average_cell; i++)
		{
			y_av_new_x[i] = y_av_x[i] + tau * Psi_av_new_x[i];
			y_av_new_y[i] = y_av_y[i] + tau * Psi_av_new_y[i];
			y_av_new_z[i] = y_av_z[i] + tau * Psi_av_new_z[i];

			x_av_new_x_1[i] = -1.0 * b_cell_1[i] * beta_cell[i] * (x_av_x_1[i] + tau * Psi_av_new_x[i]) / (b_cell_1[i] * b_cell_1[i]);
			x_av_new_y_1[i] = -1.0 * b_cell_1[i] * beta_cell[i] * (x_av_y_1[i] + tau * Psi_av_new_y[i]) / (b_cell_1[i] * b_cell_1[i]);
			x_av_new_z_1[i] = -1.0 * b_cell_1[i] * beta_cell[i] * (x_av_z_1[i] + tau * Psi_av_new_z[i]) / (b_cell_1[i] * b_cell_1[i]);

			x_av_new_x_1[i] = x_av_new_x_1[i] + (x_av_x_1[i] + tau * Psi_av_new_x[i]) / (b_cell_1[i] * b_cell_1[i]) + (x_av_x_2[i] + tau * Psi_av_new_x[i]) / (b_cell_1[i] * b_cell_2[i]);
			x_av_new_y_1[i] = x_av_new_y_1[i] + (x_av_y_1[i] + tau * Psi_av_new_y[i]) / (b_cell_1[i] * b_cell_1[i]) + (x_av_y_2[i] + tau * Psi_av_new_y[i]) / (b_cell_1[i] * b_cell_2[i]);
			x_av_new_z_1[i] = x_av_new_z_1[i] + (x_av_z_1[i] + tau * Psi_av_new_z[i]) / (b_cell_1[i] * b_cell_1[i]) + (x_av_z_2[i] + tau * Psi_av_new_z[i]) / (b_cell_1[i] * b_cell_2[i]);

			x_av_new_x_1[i] = x_av_new_x_1[i] * (-1.0 * t_stop_average_1[i]) / (tau * eps_cell_1[i] * beta_cell[i]);
			x_av_new_y_1[i] = x_av_new_y_1[i] * (-1.0 * t_stop_average_1[i]) / (tau * eps_cell_1[i] * beta_cell[i]);
			x_av_new_z_1[i] = x_av_new_z_1[i] * (-1.0 * t_stop_average_1[i]) / (tau * eps_cell_1[i] * beta_cell[i]);



			x_av_new_x_2[i] = -1.0 * b_cell_2[i] * beta_cell[i] * (x_av_x_2[i] + tau * Psi_av_new_x[i]) / (b_cell_2[i] * b_cell_2[i]);
			x_av_new_y_2[i] = -1.0 * b_cell_2[i] * beta_cell[i] * (x_av_y_2[i] + tau * Psi_av_new_y[i]) / (b_cell_2[i] * b_cell_2[i]);
			x_av_new_z_2[i] = -1.0 * b_cell_2[i] * beta_cell[i] * (x_av_z_2[i] + tau * Psi_av_new_z[i]) / (b_cell_2[i] * b_cell_2[i]);

			x_av_new_x_2[i] = x_av_new_x_2[i] + (x_av_x_1[i] + tau * Psi_av_new_x[i]) / (b_cell_2[i] * b_cell_1[i]) + (x_av_x_2[i] + tau * Psi_av_new_x[i]) / (b_cell_2[i] * b_cell_2[i]);
			x_av_new_y_2[i] = x_av_new_y_2[i] + (x_av_y_1[i] + tau * Psi_av_new_y[i]) / (b_cell_2[i] * b_cell_1[i]) + (x_av_y_2[i] + tau * Psi_av_new_y[i]) / (b_cell_2[i] * b_cell_2[i]);
			x_av_new_z_2[i] = x_av_new_z_2[i] + (x_av_z_1[i] + tau * Psi_av_new_z[i]) / (b_cell_2[i] * b_cell_1[i]) + (x_av_z_2[i] + tau * Psi_av_new_z[i]) / (b_cell_2[i] * b_cell_2[i]);

			x_av_new_x_2[i] = x_av_new_x_2[i] * (-1.0 * t_stop_average_2[i]) / (tau * eps_cell_2[i] * beta_cell[i]);
			x_av_new_y_2[i] = x_av_new_y_2[i] * (-1.0 * t_stop_average_2[i]) / (tau * eps_cell_2[i] * beta_cell[i]);
			x_av_new_z_2[i] = x_av_new_z_2[i] * (-1.0 * t_stop_average_2[i]) / (tau * eps_cell_2[i] * beta_cell[i]);

		}

		for (i = 0; i <= Number_of_average_cell; i++)
		{
			tmp1 = 1.0;
			tmp1 = tmp1 + eps_cell_1[i] + eps_cell_2[i];

			Vx_g_average[i] = y_av_new_x[i];
			Vy_g_average[i] = y_av_new_y[i];
			Vz_g_average[i] = y_av_new_z[i];



			Vx_g_average[i] = Vx_g_average[i] + eps_cell_1[i] * x_av_new_x_1[i];
			Vy_g_average[i] = Vy_g_average[i] + eps_cell_1[i] * x_av_new_y_1[i];
			Vz_g_average[i] = Vz_g_average[i] + eps_cell_1[i] * x_av_new_z_1[i];

			Vx_d_average_1[i] = y_av_new_x[i];
			Vy_d_average_1[i] = y_av_new_y[i];
			Vz_d_average_1[i] = y_av_new_z[i];

			Vx_d_average_1[i] = Vx_d_average_1[i] + eps_cell_2[i] * x_av_new_x_2[i];
			Vy_d_average_1[i] = Vy_d_average_1[i] + eps_cell_2[i] * x_av_new_y_2[i];
			Vz_d_average_1[i] = Vz_d_average_1[i] + eps_cell_2[i] * x_av_new_z_2[i];

			Vx_d_average_1[i] = Vx_d_average_1[i] - (tmp1 - eps_cell_1[i]) * x_av_new_x_1[i];
			Vy_d_average_1[i] = Vy_d_average_1[i] - (tmp1 - eps_cell_1[i]) * x_av_new_y_1[i];
			Vz_d_average_1[i] = Vz_d_average_1[i] - (tmp1 - eps_cell_1[i]) * x_av_new_z_1[i];

			Vx_d_average_1[i] = Vx_d_average_1[i] / tmp1;
			Vy_d_average_1[i] = Vy_d_average_1[i] / tmp1;
			Vz_d_average_1[i] = Vz_d_average_1[i] / tmp1;


			Vx_g_average[i] = Vx_g_average[i] + eps_cell_2[i] * x_av_new_x_2[i];
			Vy_g_average[i] = Vy_g_average[i] + eps_cell_2[i] * x_av_new_y_2[i];
			Vz_g_average[i] = Vz_g_average[i] + eps_cell_2[i] * x_av_new_z_2[i];

			Vx_d_average_2[i] = y_av_new_x[i];
			Vy_d_average_2[i] = y_av_new_y[i];
			Vz_d_average_2[i] = y_av_new_z[i];

			Vx_d_average_2[i] = Vx_d_average_2[i] + eps_cell_1[i] * x_av_new_x_1[i];
			Vy_d_average_2[i] = Vy_d_average_2[i] + eps_cell_1[i] * x_av_new_y_1[i];
			Vz_d_average_2[i] = Vz_d_average_2[i] + eps_cell_1[i] * x_av_new_z_1[i];

			Vx_d_average_2[i] = Vx_d_average_2[i] - (tmp1 - eps_cell_2[i]) * x_av_new_x_2[i];
			Vy_d_average_2[i] = Vy_d_average_2[i] - (tmp1 - eps_cell_2[i]) * x_av_new_y_2[i];
			Vz_d_average_2[i] = Vz_d_average_2[i] - (tmp1 - eps_cell_2[i]) * x_av_new_z_2[i];

			Vx_d_average_2[i] = Vx_d_average_2[i] / tmp1;
			Vy_d_average_2[i] = Vy_d_average_2[i] / tmp1;
			Vz_d_average_2[i] = Vz_d_average_2[i] / tmp1;

			Vx_g_average[i] = Vx_g_average[i] / tmp1;
			Vy_g_average[i] = Vy_g_average[i] / tmp1;
			Vz_g_average[i] = Vz_g_average[i] / tmp1;
		}

	/*	out_file = fopen("Data/Cell_av_new.dat", "wt");
		fprintf(out_file, "t=%5.3f \n", Tm);
		for (i = 0; i <= Number_of_average_cell; i++)
		{
			fprintf(out_file, "%d \t %d \t %10.8lf \t %10.8lf \t %10.8lf \t %d \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \n",
				i, g_average_count[i], Vx_g_average[i], Vy_g_average[i], Vz_g_average[i], d_average_count_1[i], Vx_d_average_1[i], Vy_d_average_1[i], Vz_d_average_1[i], b_cell_1[i],
				d_average_count_2[i], Vx_d_average_2[i], Vy_d_average_2[i], Vz_d_average_2[i], b_cell_2[i]);

		}
		fclose(out_file);
		*/


		cudaMemcpy(dev_Vx_g_average, Vx_g_average, (Number_of_average_cell  + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vy_g_average, Vy_g_average, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vz_g_average, Vz_g_average, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);

		cudaMemcpy(dev_Vx_d_average_1, Vx_d_average_1, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vy_d_average_1, Vy_d_average_1, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vz_d_average_1, Vz_d_average_1, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_t_stop_average_1,  t_stop_average_1, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_eps_cell_1, eps_cell_1, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		
		cudaMemcpy(dev_Vx_d_average_2, Vx_d_average_2, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vy_d_average_2, Vy_d_average_2, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vz_d_average_2, Vz_d_average_2, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_t_stop_average_2, t_stop_average_2, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_eps_cell_2, eps_cell_2, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);

		V_dustKernel_IDIC << <gridSize, blockSize >> > (dev_x_dust_1, dev_y_dust_1, dev_z_dust_1, dev_Vx_dust_1, dev_Vy_dust_1, dev_Vz_dust_1, dev_Vx_g_average, dev_Vy_g_average, dev_Vz_g_average, dev_t_stop_average_1, dev_ind_dust_1,
			X_min, Y_min, Z_min, average_cell_width, Number_of_average_cell_x, Number_of_average_cell_y, Number_of_average_cell, tau, Pm);
		cudaDeviceSynchronize();

		V_dustKernel_IDIC << <gridSize, blockSize >> > (dev_x_dust_2, dev_y_dust_2, dev_z_dust_2, dev_Vx_dust_2, dev_Vy_dust_2, dev_Vz_dust_2, dev_Vx_g_average, dev_Vy_g_average, dev_Vz_g_average, dev_t_stop_average_2, dev_ind_dust_2,
			X_min, Y_min, Z_min, average_cell_width, Number_of_average_cell_x, Number_of_average_cell_y, Number_of_average_cell, tau, Pm);


		cudaMemcpy(Vx_dust_1, dev_Vx_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vy_dust_1, dev_Vy_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vz_dust_1, dev_Vz_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		
		cudaMemcpy(Vx_dust_2, dev_Vx_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vy_dust_2, dev_Vy_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vz_dust_2, dev_Vz_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);

		cudaMemcpy(dev_Psi_av_new_x, Psi_av_new_x, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Psi_av_new_y, Psi_av_new_y, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Psi_av_new_z, Psi_av_new_z, (Number_of_average_cell + 1) * sizeof(double), cudaMemcpyHostToDevice);

		V_gasKernel_IDIC << <gridSize, blockSize >> > (dev_x_gas, dev_y_gas, dev_z_gas, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_Vx_d_average_1, dev_Vy_d_average_1, dev_Vz_d_average_1, dev_Vx_d_average_2, dev_Vy_d_average_2, dev_Vz_d_average_2,
			dev_Psi_av_new_x, dev_Psi_av_new_y, dev_Psi_av_new_z, dev_t_stop_average_1, dev_t_stop_average_2, dev_eps_cell_1, dev_eps_cell_2, dev_ind_gas, X_min, Y_min, Z_min, average_cell_width, Number_of_average_cell_x, Number_of_average_cell_y, Number_of_average_cell, tau, Pm);
		cudaDeviceSynchronize();

		cudaMemcpy(Vx_gas, dev_Vx_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vy_gas, dev_Vy_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vz_gas, dev_Vz_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);


		MoveKernel_IDIC << <gridSize, blockSize >> > (dev_x_gas, dev_y_gas, dev_z_gas, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_ind_gas, tau, Pm);
		cudaDeviceSynchronize();

		MoveKernel_IDIC << <gridSize, blockSize >> > (dev_x_dust_1, dev_y_dust_1, dev_z_dust_1, dev_Vx_dust_1, dev_Vy_dust_1, dev_Vz_dust_1, dev_ind_dust_1, tau, Pm);
		cudaDeviceSynchronize();

		MoveKernel_IDIC << <gridSize, blockSize >> > (dev_x_dust_2, dev_y_dust_2, dev_z_dust_2, dev_Vx_dust_2, dev_Vy_dust_2, dev_Vz_dust_2, dev_ind_dust_2, tau, Pm);
		cudaDeviceSynchronize();

		cudaMemcpy(x_gas, dev_x_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(y_gas, dev_y_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(z_gas, dev_z_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		
		cudaMemcpy(x_dust_1, dev_x_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(y_dust_1, dev_y_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(z_dust_1, dev_z_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		

		cudaMemcpy(x_dust_2, dev_x_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(y_dust_2, dev_y_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(z_dust_2, dev_z_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);

		
		EnergyKernel_IDIC << <gridSize, blockSize >> > (dev_e_gas, dev_e_temp, dev_ind_gas, tau, Pm);
		cudaDeviceSynchronize();

		cudaMemcpy(e_gas, dev_e_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		



		if (Tm >= out_num * T_out)
		{

			cudaMemcpy(x_gas, dev_x_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(y_gas, dev_y_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(z_gas, dev_z_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(rho_gas, dev_rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(p_gas, dev_p_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(e_gas, dev_e_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vx_gas, dev_Vx_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vy_gas, dev_Vy_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vz_gas, dev_Vz_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);

			cudaMemcpy(x_dust_1, dev_x_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(y_dust_1, dev_y_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(z_dust_1, dev_z_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vx_dust_1, dev_Vx_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vy_dust_1, dev_Vy_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vz_dust_1, dev_Vz_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(rho_dust_1, dev_rho_dust_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(t_stop_1, dev_t_stop_1, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);


			cudaMemcpy(x_dust_2, dev_x_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(y_dust_2, dev_y_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(z_dust_2, dev_z_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vx_dust_2, dev_Vx_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vy_dust_2, dev_Vy_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(Vz_dust_2, dev_Vz_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(rho_dust_2, dev_rho_dust_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
			cudaMemcpy(t_stop_2, dev_t_stop_2, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);


			Data_out(out_num);
			Data_out_dust(out_num);
			out_num = out_num + 1;
		}

	} while (Tm < T_end);

	/*
	do
	{
		Tm = Tm + tau;
		printf("Time %5.3lf \n", Tm);


		// Allocation particled to cells

		for (i = 0; i <= Cl; i++)
		{
			Cell[i] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			Nn[p] = -1;
		}

		for (p = 0; p <= Pm; p++)
		{
			i = int((x_gas[p] - Clx) / Clh) + (int((y_gas[p] - Cly) / Clh)) * Cnx + (int((z_gas[p] - Clz) / Clh)) * Cnx * Cny;
			if ((i >= 0) && (i <= Cl))
			{
				if (Cell[i] == -1) { Cell[i] = p; Ne[i] = p; Nn[p] = -1; }
				else { Nn[Ne[i]] = p; Ne[i] = p; Nn[p] = -1; };
			}
		}


		cudaMemcpy(dev_x_gas, x_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_y_gas, y_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_z_gas, z_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_rho_gas, rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_p_gas, p_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_mas_gas, mas_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_e_gas, e_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vx_gas, Vx_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vy_gas, Vy_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Vz_gas, Vz_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Ax_gas, Ax_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Ay_gas, Ay_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Az_gas, Az_gas, (Pm + 1) * sizeof(double), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Ind_gas, Ind_gas, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Cell, Cell, (Cl) * sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(dev_Nn, Nn, (Pm + 1) * sizeof(int), cudaMemcpyHostToDevice);
		
		RhoInitKernel <<<gridSize, blockSize >>> (dev_rho_gas, dev_Ind_gas, Pm);
		cudaDeviceSynchronize();

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{

					RhoKernel <<<gridSize, blockSize >>> (dev_x_gas, dev_y_gas, dev_z_gas, dev_rho_gas, dev_mas_gas, dev_Ind_gas, dev_Cell, dev_Nn, Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Cl, h, k, l, g);
					cudaDeviceSynchronize();
				}

		PKernel <<<gridSize, blockSize >>> (dev_rho_gas, dev_p_gas, dev_e_gas, dev_Ind_gas, Pm, Type_of_state, Gam_g, Speed_of_sound_gas);
		cudaDeviceSynchronize();


		ForceInitKernel <<<gridSize, blockSize >>> (dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_Ind_gas, dev_e_temp, Pm);
		cudaDeviceSynchronize();

		for (k = -1; k <= 1; k++)
			for (l = -1; l <= 1; l++)
				for (g = -1; g <= 1; g++)
				{
					ForceKernel <<<gridSize, blockSize >>> (dev_x_gas, dev_y_gas, dev_z_gas, dev_rho_gas, dev_p_gas, dev_mas_gas, dev_e_gas, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_e_temp, dev_Ind_gas, dev_Cell, dev_Nn, Pm, Clx, Cly, Clz, Clh, Cnx, Cny, Cl, h, tau, Gam_g, alpha, beta, eps, k, l, g);
					cudaDeviceSynchronize();
				}


		MoveKernel <<<gridSize, blockSize >>> (dev_x_gas, dev_y_gas, dev_z_gas, dev_Vx_gas, dev_Vy_gas, dev_Vz_gas, dev_Ax_gas, dev_Ay_gas, dev_Az_gas, dev_Ind_gas, dev_e_temp, dev_e_gas, tau, Pm);
		cudaDeviceSynchronize();



		cudaMemcpy(x_gas, dev_x_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(y_gas, dev_y_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(z_gas, dev_z_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(rho_gas, dev_rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(p_gas, dev_p_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(mas_gas, dev_mas_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(e_gas, dev_e_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vx_gas, dev_Vx_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vy_gas, dev_Vy_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Vz_gas, dev_Vz_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ax_gas, dev_Ax_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Ay_gas, dev_Ay_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(Az_gas, dev_Az_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		
		
		if (Tm >= out_num * T_out)
		{
			Data_out(out_num);
			out_num = out_num + 1;
		}

	} while (Tm < T_end);
	
	*/

    return 0;
}

