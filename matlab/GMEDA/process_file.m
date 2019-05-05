function [X,Y,indicator,n,m] = process_file(data_address,domains,source_domain_num)
    X = [];
    Y = [];
    indicator = [];
    ed = 0;
    n = 0;
    m = 0;
    for i = 1:4
        if i<source_domain_num + 1
            n = n + 1;
        else
            m = m + 1;
        end
        src = domains{i};
        load([data_address src '_SURF_L10.mat']);     % source domain
        fts = fts ./ repmat(sum(fts,2),1,size(fts,2)); 
        num = size(fts,1);
        %Z = zscore(X) returns the z-score for each element of X such that columns of X are centered to have mean 0 and scaled to have standard deviation 1. Z is the same size as X.
        %Indicator for the standard deviation used to compute the z-scores, specified as 0 or 1.
        %If flag is 0 (default), then zscore scales X using the sample standard deviation. zscore(X,0) is the same as zscore(X).
        %If flag is 1, then zscore scales X using the population standard deviation.
        X = [X;zscore(fts,1)];    clear fts
        Y = [Y;labels];           clear labels
        indicator = [indicator;[ed+1,ed+num,i]];
        ed = ed + num;
    end
end