double q2min(double x){
  double q2 = 0.511e-3*0.511e-3*x*x/(1-x);
  if (q2 >= 1) return 1;
  else return q2;
}

double f(double x){
  return 1/(2*3.1416*137)*( (1 + (1-x)*(1-x))/x * log(1/q2min(x)) + 2*0.511e-3*0.511e-3*x*(1 - 1/q2min(x)) );
}

void eff_photon_lumi(){
  int n=100;
  TCanvas *c = new TCanvas();
  for(int i=0; i<n; i++){
    double w = 10*pow(1.0471285, -i);
    double z = w/10.02;
    auto d2l_over_dw_dx = [z](double *x, double *par){ return 2*z*f(x[0])*f(z*z/x[0])/x[0]/10.02; };
    TF1 *func = new TF1("f", d2l_over_dw_dx, 0, 1, 0);
    func->Draw();
    double int_f = func->Integral(z*z, 1);
    cout << w << " " << int_f << endl;
  }
}
