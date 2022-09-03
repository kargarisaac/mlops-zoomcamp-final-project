from zenml.pipelines import pipeline
from zenml.steps import Output, step
import random


@step
def get_first_num() -> Output(first_num=int):
    """Returns an integer."""
    print("inside the first step!")
    return 10

@step(enable_cache=False)
def get_random_int() -> Output(random_num=int):
    """Get a random integer between 0 and 10."""
    print("inside the second step!")
    return random.randint(0, 10)

@step
def subtract_numbers(first_num: int, random_num: int) -> Output(result=int):
    """Subtract random_num from first_num."""
    print("inside the third step!")
    return first_num - random_num

@pipeline
def vertex_example_pipeline(first_step, second_step, third_step):
    # Link all the steps artifacts together
    first_num = first_step()
    random_num = second_step()
    third_step(first_num, random_num)


if __name__ == "__main__":
    # Initialize a new pipeline run
    pipeline_instance = vertex_example_pipeline(
        first_step=get_first_num(),
        second_step=get_random_int(),
        third_step=subtract_numbers(),
    )

    # Run the new pipeline
    pipeline_instance.run()
