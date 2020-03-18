
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
	if (k < 1.0) { tmp1 = (double)1.0 - 1.5 * k * k + 0.75 * k * k * k; }
	if ((k >= 1.0) && (k <= 2.0)) { tmp1 = (double)0.25 * (2 - k) * (2 - k) * (2 - k); }
	if (k > 2.0) { tmp1 = 0.0; }

	return  c / (h * h * h) * tmp1;
}

__device__ __host__ double dW(double r, double h)
{
	const double Pi = 3.14159265;
	double k, c, tmp1;
	k = (double) r / h;
	c = (double)1 / (Pi);
	if (k < -2.0) { tmp1 = 0.0; }
	if ((k >= -2.0) && (k <= -1.0)) { tmp1 = (double)0.75 * (2.0 + k) * (2.0 + k); }
	if ((k > -1.0) && (k < 0)) { tmp1 = (double)-3.0 * k - 2.25 * k * k; }
	if ((k >= 0) && (k <= 1.0)) { tmp1 = (double)-3.0 * k + 2.25 * k * k; }
	if ((k >= 1.0) && (k <= 2.0)) { tmp1 = (double)-0.75 * (2.0 - k) * (2.0 - k); }
	if (k > 2.0) { tmp1 = 0.0; }

	return  c / (h * h * h * h) * tmp1;
}

double Eq_State(double rho, double e)
{
	//	return B*(pow(rho,Gam_liq)-1) + 1;
	if (Type_of_state == 0) { return rho * e * (Gam_g - 1); }
	if (Type_of_state == 1) { return Speed_of_sound_gas * Speed_of_sound_gas * rho; }
}

double Eq_State1(double p, double e)
{
	//	return pow((p-1)/B+1,1.0/Gam_liq);

	if (Type_of_state == 0) { return p / (e * (Gam_g - 1)); }
	if (Type_of_state == 1) { return p / (Speed_of_sound_gas * Speed_of_sound_gas); }
}


int main()
{
	double dlh;

	FILE *ini_file;
	char s[128];
	fopen_s(&ini_file, "Init.txt", "r");
	fgets(s, 128, ini_file);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &X_min);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &X_max);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Y_min);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Y_max);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Z_min);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Z_max);
	fgets(s, 128, ini_file); sscanf(s, "%d", &Maximum_particle);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Particle_on_length);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &h);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &tau);


/*	fgets(s, 128, ini_file);	sscanf(s, "%lf", &Zm);
	fgets(s, 128, ini_file);	sscanf(s, "%lf", &c0);
	fgets(s, 128, ini_file);	sscanf(s, "%lf", &n);
	
	fgets(s, 128, ini_file); sscanf(s, "%lf", &p_in);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &p_out);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &tau);
	fgets(s, 128, ini_file);	sscanf(s, "%d", &Te);
	fgets(s, 128, ini_file);	sscanf(s, "%d", &File_int);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &Geps);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &alfa);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &beta);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &eps);
	fgets(s, 128, ini_file); sscanf(s, "%lf", &sigm);	
*/

	fclose(ini_file);

	Type_of_state = 0;
	dlh = (double) 1.0 / Particle_on_length;
	Tm = 0.0;
	out_num = 0;

	// Memory

	x_gas = new double[Maximum_particle];
	y_gas = new double[Maximum_particle];
	z_gas = new double[Maximum_particle];
	p_gas = new double[Maximum_particle];
	rho_gas = new double[Maximum_particle];
	Vx_gas = new double[Maximum_particle];
	Vy_gas = new double[Maximum_particle];
	Vz_gas = new double[Maximum_particle];
	Ind_gas = new int[Maximum_particle];
	

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

	Cl = (Cnx + 1) * (Cny + 1) * (Cnz + 1);
	Cell = new int[Cl + 1]; // Number of first particle
	Ne = new int[Cl + 1]; //  Number of last particle



	// Particle placing

	p = -1;
	Im = int ((X_max - X_min) * Particle_on_length);
	Jm = int ((Y_max - Y_min) * Particle_on_length);
	Km = int ((Z_max - Z_min) * Particle_on_length); 

	// Среда

	double x_temp, y_temp, z_temp;

	for (i = 0; i <= Im; i++)
		for (j = 0; j <= Jm; j++)
			for (k = 0; k <= Km; k++)
			{
				x_temp = X_min + (double) (i * dlh);
				y_temp = Y_min + (double) (j * dlh);
				z_temp = Z_min + (double) (k * dlh);

				p = p + 1;
				x_gas[p] = x_temp;// + (rand()%100-50.0)/100000.0 * dlh;
				y_gas[p] = y_temp;// + (rand()%100-50.0)/100000.0 * dlh;
				z_gas[p] = z_temp;// + (rand()%100-50.0)/100000.0 * dlh;
				Vx_gas[p] = 0;
				Vy_gas[p] = 0;
				Vz_gas[p] = 0;
				Ind_gas[p] = 0;
			}

	Pr = p;

	Pm = p;

	do
	{

	} while (Tm < T_end);

    return 0;
}

