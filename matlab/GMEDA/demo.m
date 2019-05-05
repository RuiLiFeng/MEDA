% DEMO for testing GMEDA on Office+Caltech10 datasets
str_domains = {'Caltech10', 'amazon', 'webcam', 'dslr'};
data_address = '/gdata/fengruili/MEDA/surf/';
list_acc = [];

for i = 1 : 4
    domains = rearrange(str_domains,i);
    [X,Y,indicator,n,m] = process_file(data_address,domains,source_domain_num);
    options = set_option(20,1.0,10,10.0,0.1,10);
    [Acc,~,~,~] = MEDA(X,Y,indicator,n,m,options);
    fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
end

function str = rearrange(domains,i)
    %target on the bottom
    str = [domains(1:i-1),domains(i+1:end),domains(i)];
end

function options = set_option(d,rho,p,lambda,eta,T)
    options.d = d;
    options.rho = rho;
    options.p = p;
    options.lambda = lambda;
    options.eta = eta;
    options.T = T;
end