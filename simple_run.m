% simple run 
% X=200 x 200, r=10
% poisson noise
% Note that all zero columns and rows of the input must be removed

clear all; close all; clc; 

m =100;
n = 100;
max_iter=100000;
max_time=4;
r=10;
options.timemax =max_time;
options.maxiter=max_iter;
Vtrue = rand(r,n); 
Utrue = rand(m,r); 
X= Utrue*Vtrue; 

% if you want to add possion noise: 
% X= poissrnd(X); 

colX=sum(X,2)/n+eps;
nX=X.*log(X./repmat(colX,1,n)+eps);
nX=sum(nX(:));

% initialization
W= rand(m,r);
H = rand(r,n);
% scale initial point 
 WH=W*H;
 alpha=sqrt(sum(X(:))/sum(WH(:)));
 W=alpha*W; 
 H=alpha*H; 
 options.init.W=W; 
 options.init.H=H;
 
 % run algorithms 
 % NoLips
[W_BMD,H_BMD,e_Nl,t_Nl] = BMD(X,r,options) ;
e_Nl=e_Nl/nX; % relative error 
fprintf('... BMD done, final error = %f \n',e_Nl(end));

% Scalar Newton
[W_tfPNcol, H_fPNcol, e_fPNcol ,t_fPNcol]=SN(X,r,max_iter,max_time,W',H,5,0.2);
 e_fPNcol = e_fPNcol /nX;
fprintf('... Scalar Newton done, final error = %f \n',e_fPNcol(end));

%    MU  
 [Wmu,Hmu,e_MU,t_MU]=MU(X,r,options);
 e_MU=e_MU/nX; 
 fprintf('... MU done, final error = %f \n',e_MU(end));
 
 % CCD   
[w0, h0, e_CCD, t_CCD] = KLnmf(X,r,max_iter,max_time,W',H, 1);
e_CCD=e_CCD/nX;
fprintf('... CCD done, final error = %f \n',e_CCD(end));

%ADMM
beta=1; 
[Wad,Had,e_admm,t_admm]=nmf_admm(X,beta,1,options);
e_admm=e_admm/nX; 
 fprintf('... ADMM done, final error = %f \n',e_admm(end));

%primal-dual
  D=5;N = 1e3;
 [W_PD, H_PD, e_pd, t_pd] = nmf_kl_fpa(X, W, H, N, D,options);
 e_pd=e_pd/nX;
  fprintf('... PD done, final error = %f \n',e_pd(end));

% SN-MU
 [W_SNMU,H_SNMU,e_SNMU,t_SNMU]=SN_MU(X,r,options);
 e_SNMU=e_SNMU/nX;
 fprintf('... SN-MU done, final error = %f \n',e_SNMU(end));

e_min=min([min(e_SNMU),min(e_fPNcol),min(e_MU),min(e_CCD),min(e_admm),min(e_pd),min(e_Nl)]);
figure;
set(0, 'DefaultAxesFontSize', 18);
set(0, 'DefaultLineLineWidth', 2);

%semilogy(xk,Fk,'Color',[0.53,0.33,0.65],'Marker','v','LineWidth',2)

semilogy(t_MU,e_MU-e_min,'g-.','LineWidth',1.5);hold on; %MU
semilogy(t_fPNcol,e_fPNcol-e_min,'k','LineWidth',3);hold on; 
semilogy(t_CCD,e_CCD-e_min,'y--','LineWidth',3);hold on; 
semilogy(t_admm,e_admm-e_min,'b','LineWidth',1.5);hold on; 
semilogy(t_pd,e_pd-e_min,'m-','LineWidth',1.5);hold on; 
semilogy(t_Nl,e_Nl-e_min,'c--','LineWidth',3);hold on; 
semilogy(t_SNMU,e_SNMU-e_min,'r-.','LineWidth',3);hold on; 
ylabel('rel D(X,WH) - e_{min}');
xlabel('Time')
legend('MU','SN','CCD','ADMM','PD','BMD','SN-MU'); 

