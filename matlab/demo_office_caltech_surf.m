% DEMO for testing MEDA on Office+Caltech10 datasets
str_domains = {'Caltech10', 'amazon', 'webcam', 'dslr'};
data_address = '/gdata/fengruili/MEDA/surf/';
list_acc = [];
for i = 1 : 4
    for j = 1 : 4
        if i == j
            continue;
        end
        src = str_domains{i};
        tgt = str_domains{j};
        load([data_address src '_SURF_L10.mat']);     % source domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        %Z = zscore(X) returns the z-score for each element of X such that columns of X are centered to have mean 0 and scaled to have standard deviation 1. Z is the same size as X.
        %Indicator for the standard deviation used to compute the z-scores, specified as 0 or 1.
        %If flag is 0 (default), then zscore scales X using the sample standard deviation. zscore(X,0) is the same as zscore(X).
        %If flag is 1, then zscore scales X using the population standard deviation.
        
        Xs = zscore(fts,1);    clear fts
        Ys = labels;           clear labels
        
        load([data_address tgt '_SURF_L10.mat']);     % target domain
        %B = repmat(A,r1,...,rN) 指定一个标量列表 r1,..,rN，这些标量用于描述 A 的副本在每个维度中如何排列。当 A 具有 N 维时，B 的大小为 size(A).*[r1...rN]。例如：repmat([1 2; 3 4],2,3) 返回一个 4×6 的矩阵。
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        Xt = zscore(fts,1);     clear fts
        Yt = labels;            clear labels
        
        % meda
        options.d = 20;
        options.rho = 1.0;
        options.p = 10;
        options.lambda = 10.0;
        options.eta = 0.1;
        options.T = 10;
        [Acc,~,~,~] = MEDA(Xs,Ys,Xt,Yt,options);
        fprintf('%s --> %s: %.2f accuracy \n\n', src, tgt, Acc * 100);
    end
end
