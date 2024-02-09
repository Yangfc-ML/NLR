load('birds_r=3 sample.mat');

optmparameter.beta=100;
optmparameter.gamma=10;
optmparameter.lambda=2.1;
% for data set music_style, lambda=1.7

threshold=0.9;
result=zeros(5,5);

for n =1:5
    % load data
    test_data = data(te_idx{n},:);
    test_target=target(:,te_idx{n})';
    train_data=data(tr_idx{n},:);
    train_p_target=candidate_labels(:,tr_idx{n})';
    % training
    [W1,W2,N]=NLR(train_data,train_p_target,optmparameter);
    % testing
    predict_values=test_data*W1;
    [num_testing,num_class] = size(test_target);
    outputValue=predict_values;
    for i=1:num_testing
        outputValue(i,:)=(outputValue(i,:)-min(min(outputValue(i,:))))/(max(max(outputValue(i,:)))-min(min(outputValue(i,:))));
    end
    Outputs=outputValue';
    Pre_Labels=zeros(num_class,num_testing);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    for i=1:num_testing
        for j=1:num_class
            if(Outputs(j,i)>threshold)
                Pre_Labels(j,i)=1;
            else
                Pre_Labels(j,i)=0;
            end
        end
    end
    HammingLoss=Hamming_loss(Pre_Labels,test_target');
    RankingLoss=Ranking_loss(Outputs,test_target');
    OneError=One_error(Outputs,test_target');
    Coverage=coverage(Outputs,test_target');
    Average_Precision=Average_precision(Outputs,test_target');
    result(n,:) = [HammingLoss,RankingLoss,OneError,Coverage,Average_Precision];
end
HammingLoss=result(:,1);
RankingLoss=result(:,2);
OneError=result(:,3);
Coverage=result(:,4);
AveragePrecision=result(:,5);

fprintf('NLR HammingLoss: %f std: %f\nNLR RankingLoss: %f std: %f\nNLR OneError: %f std: %f\nNLR Coverage: %f std: %f\nNLR Average_Precision: %f std: %f\n',mean(HammingLoss),std(HammingLoss),mean(RankingLoss),std(RankingLoss),mean(OneError),std(OneError),mean(Coverage),std(Coverage),mean(AveragePrecision),std(AveragePrecision));

