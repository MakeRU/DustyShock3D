
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

void Data_out(int num)
{
	FILE* out_file_gas, * out_file_dust, * out_file;
	char out_name[25];
	int i, j, l;


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
	fprintf(out_file_gas, "x \t mas \t rho \t p \t Vx \t Vy \t Vz \t Ax \t Ay \t Az \t e \t Ind \n");

	for (i = 0; i <= Pm; i++)
	{
		if ((x_gas[i] >= 0.0) && (x_gas[i] <= 1.0))
		{
			fprintf(out_file_gas, "%10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %10.8lf \t %d \n",
				x_gas[i], mas_gas[i], rho_gas[i], p_gas[i], Vx_gas[i], Vy_gas[i], Vy_gas[i], Ax_gas[i], Ay_gas[i], Az_gas[i], e_gas[i], Ind_gas[i]);
		}

	}

	fclose(out_file_gas);

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
	fgets(s, 128, ini_file); sscanf(s, "%lf", &tau);
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
	mas_gas = new double[Maximum_particle];
	Vx_gas = new double[Maximum_particle];
	Vy_gas = new double[Maximum_particle];
	Vz_gas = new double[Maximum_particle];
	Ax_gas = new double[Maximum_particle];
	Ay_gas = new double[Maximum_particle];
	Az_gas = new double[Maximum_particle];
	e_gas = new double[Maximum_particle];
	Ind_gas = new int[Maximum_particle];
	Nn = new int[Maximum_particle];

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
				mas_gas[p] = 1.0 / Particle_on_length;
				Vx_gas[p] = 0.0;
				Vy_gas[p] = 0.0;
				Vz_gas[p] = 0.0;
				Ind_gas[p] = 0;
			}

	Pr = p;

	Pm = p;

	cudaSetDevice(0);

	cudaMalloc((void**)&dev_x_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_y_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_z_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Vx_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Vy_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Vz_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Ax_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Ay_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Az_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_rho_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_p_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_e_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_mas_gas, (Maximum_particle + 1) * sizeof(double));
	cudaMalloc((void**)&dev_Ind_gas, (Maximum_particle + 1) * sizeof(int));
	cudaMalloc((void**)&dev_Cell, (Cl) * sizeof(int));
	cudaMalloc((void**)&dev_Nn, (Maximum_particle + 1) * sizeof(int));
	
	
	dim3 gridSize = dim3(Pm / 256 + 1, 1, 1);
	dim3 blockSize = dim3(16, 16, 1);

	Tm = 0.0;
	out_num = 0;

	Data_out(out_num);
	out_num = out_num + 1;

	
	do
	{
		Tm = Tm + tau;
		printf("Time %5.3lf mks \n", Tm * 1e3);


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
			if ((i > 0) && (i < Cl))
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


		cudaMemcpy(x_gas, dev_x_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(y_gas, dev_y_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(z_gas, dev_z_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(rho_gas, dev_rho_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(p_gas, dev_p_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
		cudaMemcpy(mas_gas, dev_mas_gas, (Pm + 1) * sizeof(double), cudaMemcpyDeviceToHost);
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

    return 0;
}

