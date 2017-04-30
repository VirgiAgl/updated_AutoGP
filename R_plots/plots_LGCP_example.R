library(languageR)
library(ggplot2)
library(plotrix)
library(plyr)
library(party)
library(gridExtra)
library(Hmisc)
library(extrafont)
library(scales)
library(reshape)
library(pbkrtest)
library(nloptr)
library(optimx)
library(data.table)
library(extrafont)
loadfonts()

#some helper functions
source('Workspace/updated_AutoGP/helper.R')

SP_name = "SF"

w = 15
h = 5

output_path = "../../first_pictures"

# mining data
name = 'synthetic_data'
inputs = read.csv('Workspace/updated_AutoGP/data_inputs.csv')
outputs = read.csv('Workspace/updated_AutoGP/data_outputs.csv')
posterior_mean_intensity = read.csv('Workspace/updated_AutoGP/total_results_ypred.csv')
posterior_var_intensity = read.csv('Workspace/updated_AutoGP/total_results_postvar.csv')
idx = read.csv('Workspace/updated_AutoGP/idx.csv')
xtest = read.csv('Workspace/updated_AutoGP/xtest.csv')

N = 200
idx = c(idx)
idx = as.integer(unlist(idx))
input_indices = sort(idx[(N):length(idx)])
inputs$id = c(1:nrow(inputs))
xtest = inputs[inputs$id %in% input_indices,]

#data$model = toupper(substr(data$model_sp,0, 4))
#data = rename_model(data)
#data$sp = paste(SP_name, "=", substr(data$model_sp,6, 8))

data_intensity = cbind(xtest, posterior_mean_intensity, posterior_var_intensity)
data_intensity = data_intensity[,c(1,3,4)]
colnames(data_intensity) = c("x","m","v")

model = "FG"
sp = 1
y_lab = "intensity"
#p2 = draw_intensity(data_intensity, "intensity")
ymax=data_intensity$m + 2 * sqrt(data_intensity$v)
ymin= data_intensity$m - 2 * sqrt(data_intensity$v)

p2 = ggplot(data_intensity, aes(x=x, y = m, colour = 1)) + 
  geom_point() +
  geom_ribbon(aes(x=x, ymin= m - 2 * sqrt(v), ymax=m + 2 * sqrt(v)), fill="grey", alpha=.4, colour =NA) +  
  xlab('') +
  ylab(y_lab) +
  theme_bw() +

  theme(legend.direction = "vertical", legend.position = "none", legend.box = "vertical", 
        axis.line = element_line(colour = "black"),
        panel.grid.minor=element_blank(),      
        panel.border = element_blank(),
        panel.spacing = unit(.4, "lines"),
        text=element_text(family="Arial", size=10),
        legend.key = element_blank(),
        strip.background = element_rect(colour = "white", fill = "white",
                                        size = 0.5, linetype = "solid"),
        axis.ticks.x = element_blank(),
        legend.title=element_blank(),
        axis.text.x = element_text(angle = 90, hjust = 1)
  ) 
  
data = cbind(inputs,outputs)
#data = data[,c(1,3)]
colnames(data) = c("x", "y")
p1 = draw_mining_data(data)


ggsave(file=paste(output_path, name, ".pdf", sep = ""),  width=w, height=h, units = "cm" , device=cairo_pdf, g)      


