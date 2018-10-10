#include "postgres.h"
#include "optimizer/tf.h"
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <tensorflow/c/c_api.h>

typedef struct model_t {
	TF_Graph* graph;
	TF_Session* session;
	TF_Status* status;

	TF_Output a, b, c;

} model_t;

int ModelCreate(model_t* model, const char* graph_def_filename);
void ModelDestroy(model_t* model);
int ModelInit(model_t* model);
void* ModelRun(model_t* model);

int Okay(TF_Status* status);
TF_Buffer* ReadFile(const char* filename);

int tf_run() {
	//loads the model into the model_t struct
	char* file_name = "/mnt/c/Users/abc/Documents/research/models/graph.pb";
	model_t tf_model;

	printf("Loading graph\n");
  	if (!ModelCreate(&tf_model, file_name)) return 1;

  	printf("Running graph\n");
  	void* output_result = ModelRun(&tf_model);
  	float result = *(float *)output_result;
  	printf("Output: %f\n", result);

  	ModelDestroy(&tf_model);

  	return 0;
}

int ModelCreate(model_t *model, const char* graph_def_filename) {
	model->status = TF_NewStatus();
	model->graph = TF_NewGraph();

	TF_SessionOptions* session_opts = TF_NewSessionOptions();
	model->session = TF_NewSession(model->graph, session_opts, model->status);
	TF_DeleteSessionOptions(session_opts);
	if(!Okay(model->status)) return 0;

	TF_Graph *g = model->graph;

	TF_Buffer* graph_def = ReadFile(graph_def_filename);
	if(graph_def == NULL) return 0;
	printf("Read GraphDef of %zu bytes\n", graph_def->length);
    TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
    TF_GraphImportGraphDef(g, graph_def, opts, model->status);
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(graph_def);
    if (!Okay(model->status)) return 0;

    model->a.oper = TF_GraphOperationByName(g, "a");
    model->a.index = 0;
    model->b.oper = TF_GraphOperationByName(g, "b");
    model->b.index = 0;
    model->c.oper = TF_GraphOperationByName(g, "c");
    model->c.index = 0;
}

void* ModelRun(model_t *model) {
	TF_Output inputs[2] = {model->a, model->b};

	printf("Create A Tensor\n");
	TF_Tensor* a_tensor = TF_AllocateTensor(1, NULL, 0, sizeof(float));
	float *data_a = (float *)TF_TensorData(a_tensor);
	*data_a = 4.0;

	printf("Create B Tensor\n");
	TF_Tensor* b_tensor = TF_AllocateTensor(1, NULL, 0, sizeof(float));
	float *data_b = (float *)TF_TensorData(b_tensor);
	*data_b = 3.0;

	TF_Tensor* input_values[2] = {a_tensor, b_tensor};

	TF_Output output[1] = {model->c};
	TF_Tensor* output_values[1] = {};

	printf("Running Session\n");
	TF_SessionRun(model->session, NULL, inputs, input_values, 2,
					output, output_values, 1,
					NULL, 0, NULL, model->status);

	printf("Finished Running Session\n");
	TF_DeleteTensor(a_tensor);
	TF_DeleteTensor(b_tensor);

	if(Okay(model->status)!=0) {
		return TF_TensorData(output_values[0]);
	} else{
		return NULL;
	}
}

void ModelDestroy(model_t* model) {
  TF_DeleteSession(model->session, model->status);
  Okay(model->status);
  TF_DeleteGraph(model->graph);
  TF_DeleteStatus(model->status);
}

int Okay(TF_Status* status) {
  if (TF_GetCode(status) != TF_OK) {
    fprintf(stderr, "ERROR: %s\n", TF_Message(status));
    return 0;
  }
  return 1;
}

TF_Buffer* ReadFile(const char* filename) {
  int fd = open(filename, 0);
  if (fd < 0) {
    perror("failed to open file: ");
    return NULL;
  }
  struct stat stat;
  if (fstat(fd, &stat) != 0) {
    perror("failed to read file: ");
    return NULL;
  }
  char* data = (char*)malloc(stat.st_size);
  ssize_t nread = read(fd, data, stat.st_size);
  if (nread < 0) {
    perror("failed to read file: ");
    free(data);
    return NULL;
  }
  if (nread != stat.st_size) {
    fprintf(stderr, "read %zd bytes, expected to read %zd\n", nread,
            stat.st_size);
    free(data);
    return NULL;
  }
  TF_Buffer* ret = TF_NewBufferFromString(data, stat.st_size);
  free(data);
  return ret;
}