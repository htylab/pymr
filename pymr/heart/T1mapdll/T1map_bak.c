#define EXPORT __declspec(dllexport)
#include "lmcurve.h"
#include <stdio.h>
#include <math.h>
/* model function: a parabola */

double f( double t, const double *p )
{
	return 	fabs(p[0]-p[1]*1.0*exp(-t/p[2]));
}
EXPORT void __stdcall T1map(double *t,int t_num,double *y,long y_num,double *col) 
{
    int n = 3;
    double par[3] = { 100, 200, 2500 }; 
	double guess[8] ={100,200,500,800,1000,1500,2000,2500};
	double temp[8]={0,0,0,0,0,0,0,0};
	double err[8]={0,0,0,0,0,0,0,0};
	double temp3=*t;
	double temp2=*y;
	double errd ;
	int temp4;
    int m_dat = t_num;
	int jj,ii,kk;
	double tempvalue;

    lm_control_struct control = lm_control_double;
    lm_status_struct status;
    control.verbosity = 0;
	
    for (ii=0; ii<y_num;ii++) 
	{
	for (jj=0; jj<8; jj++){
		par[0]=*(y+(ii+1)*m_dat-4);
        par[1]=par[0]*2;
		//par[2]=1000;
		par[2]=guess[jj];
		if (par[0]>10)
		{		
			lmcurve( n, par, m_dat, t, y+ii*m_dat, f, &control, &status );
			temp[jj]=(par[1]/par[0]-1)*par[2];
			errd =0;
			for (kk=0;kk<t_num;kk++)
			{

				errd+=fabs(y[kk+ii*m_dat]-f(t[kk],par));
			}
			
			err[jj] =errd;
			
			//col[ii]=(par[1]/par[0]-1)*par[2]; //T1在此
		}
		else
		{	err[jj]=1000000000;
			temp[jj]=0;
			//col[ii]=0;
		}
	//判斷暫存在temp中的八個值哪個誤差最小並存入col[ii]中		
	}
	tempvalue=err[0];
	temp4=0;	
	for (kk=1; kk<8; kk++){
		if (tempvalue > err[kk]){
			temp4 = kk;
			tempvalue = err[kk];
		}
	}	
	col[ii]=temp[temp4];
	//col[ii]=err[4];
	}
}