function [Acc,acc_iter,Beta,Yt_pred] = GMEDA(X_in,Y_in,indicator,n,m,options)

 %Input X = [Xs_1;...,Xs_ns;Xt_1;...;Xt_nt], s=nt+ns
 %X in R^(N,D),D dimension of data, N example numbers
 %indicator(i,:)=[start,end,flag]

%% Inputs:
%%% Xs      : Source domain feature matrix, n * dim
%%% Ys      : Source domain label matrix, n * 1
%%% Xt      : Target domain feature matrix, m * dim
%%% Yt      : Target domain label matrix, m * 1 (only used for testing accuracy)
%%% options : algorithm options:
%%%%% options.d      :  dimension after manifold feature learning (default: 20)
%%%%% options.T      :  number of iteration (default: 10)
%%%%% options.lambda :  lambda in the paper (default: 10)
%%%%% options.eta    :  eta in the paper (default: 0.1)
%%%%% options.rho    :  rho in the paper (default: 1.0)
%%%%% options.base   :  base classifier for soft labels (default: NN)

%% Outputs:
%%%% Acc      :  Final accuracy value
%%%% acc_iter :  Accuracy value list of all iterations, T * 1
%%%% Beta     :  Cofficient matrix
%%%% Yt_pred  :  Prediction labels for target domain

%% Algorithm starts here
    fprintf('GMEDA starts...\n');
    
    %% Load algorithm options
    if ~isfield(options,'p')
        options.p = 10;
    end
    if ~isfield(options,'eta')
        options.eta = 0.1;
    end
    if ~isfield(options,'lambda')
        options.lambda = 1.0;
    end
    if ~isfield(options,'rho')
        options.rho = 1.0;
    end
    if ~isfield(options,'T')
        options.T = 10;
    end
    if ~isfield(options,'d')
        options.d = 20;
    end
    Ys = Y_in(1:n);
    Yt = Y_in(n+1:n+m);
    % Manifold feature learning
    [X_new,L_GCA,~] = GGFK_Map(X_in,indicator,options.d,'MaxC');
    X = double(X_new');
    %to be continue...
    C = length(unique(Ys));
    acc_iter = [];
    
    YY = [];
    for c = 1 : C
        YY = [YY,Ys==c];
    end
    YY = [YY;zeros(m,C)];%size(YY)=(n+m)*C,the first m row is indicator for classes, the last m rows are zeros

    %% Data normalization
    X = X * diag(sparse(1 ./ sqrt(sum(X.^2))));%sum: sum with respect to coloumn, normalize each example

    %% Construct graph Laplacian
    if options.rho > 0
        manifold.k = options.p;
        manifold.Metric = 'Cosine';
        manifold.NeighborMode = 'KNN';
        manifold.WeightMode = 'Cosine';
        W = lapgraph(X',manifold);
        Dw = diag(sparse(sqrt(1 ./ sum(W))));
        L = eye(n + m) - Dw * W * Dw;
    else
        L = 0;
    end

    % Generate soft labels for the target domain
    knn_model = fitcknn(X(:,1:n)',Ys,'NumNeighbors',1);
    Cls = knn_model.predict(X(:,n + 1:end)');

    % Construct kernel
    K = kernel_meda('rbf',X,sqrt(sum(sum(X .^ 2).^0.5)/(n + m)));
    E = diag(sparse([ones(n,1);zeros(m,1)]));
    %Construct D
    [D_0,D_c_sum] = compute_D([Cls;Ys],indicator);
    for t = 1 : options.T
        % Estimate mu
        mu = estimate_mu(Xs',Ys,Xt',Cls);
        % Construct D matrix
        D_GCA = (1-mu)*D_0 + mu*D_c_sum;
        D_GCA = D_GCA / norm(D_GCA,'fro');


        % Compute coefficients vector Beta
        Beta = ((E + options.lambda * D_GCA*L_GCA*D_GCA' + options.rho * L) * K + options.eta * speye(n + m,n + m)) \ (E * YY);
        F = K * Beta;
        [~,Cls] = max(F,[],2);

        %% Compute accuracy
        Acc = numel(find(Cls(n+1:end)==Yt)) / m;
        Cls = Cls(n+1:end);
        acc_iter = [acc_iter;Acc];
        fprintf('Iteration:[%02d]>>mu=%.2f,Acc=%f\n',t,mu,Acc);
    end
    Yt_pred = Cls;
    fprintf('MEDA ends!\n');
end

function K = kernel_meda(ker,X,sigma)
    switch ker
        case 'linear'
            K = X' * X;
        case 'rbf'
            n1sq = sum(X.^2,1);
            n1 = size(X,2);
            D = (ones(n1,1)*n1sq)' + ones(n1,1)*n1sq -2*X'*X;
            K = exp(-D/(2*sigma^2));        
        case 'sam'
            D = X'*X;
            K = exp(-acos(D).^2/(2*sigma^2));
        otherwise
            error(['Unsupported kernel ' ker])
    end
end

function [D_0,D_c_sum] = compute_D(Y,indicator)
    C = length(unique(Y));
    s = size(indicator,1);
    num = size(Y);
    D_c_sum = zeros(num,s);
    for i = 1 : C
        D_c = [];
        for j=1:s
            st = indicator(j,2);
            ed = indicator(j,1);
            Yj = Y(st:ed);
            index = Yj == i;
            n_c = sum(Yj==i);
            D_c = blkdiag(D_c,double(index)/n_c);
        end
        D_c_sum = D_c_sum + D_c;
    end
    D_0 = [];
    for j=1:s
        st = indicator(j,2);
        ed = indicator(j,1);
        D_0 = blkdiag(D_0,ones(ed - st + 1,1)/(ed - st +1));
    end
   
end