# OLGP-code
the python implementation of information network skeleton mining algorithm
 We propose a strategy based on On-Line Graph Processing to quickly narrow down the search scope by efficiently partially materializing the information network data cube. Using this approach, we have further developed a highly efficient algorithm that automatically discovers significant substructures within subspaces at any level of multi-dimensional configurations. We explored significant substructures by drilling down from the top layer to the bottom layer of the data cube. Additionally, we designed a significant paths discovery algorithm that calculates the weight of significant paths between these structures based on the theory of maximum flow.
 The work focuses on efficiently and automatically extracting significant structures from large-scale, multi-diemensional and multi-perspective information networks. We evaluated our approach using node and edge pruning strategies on datasets from Stanford's website.
 The main programme constitutes of four parts. First, we create the tables for the upcoming process. Second, preprocess the datasets and inserts the key information into database. Third, construct the graph cube and complete the process of rolling up and drilling down operations. Fourth, discover the significant subspace from different dimensions of the graph cube and value the mining results.
 Attention that Mysql relational database is needed and connected to Python programm.
         
    
![e6a498f0083254eab2055ce3ba20bab](https://github.com/user-attachments/assets/0955b6a2-45fb-4369-a36c-ca0caadafd51)
The above figure is the overall process of our algorith.
