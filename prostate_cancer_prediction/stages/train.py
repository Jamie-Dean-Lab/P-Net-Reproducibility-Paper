def train(config):
    pipeline = config["pipeline_class"](config)
    getattr(pipeline, config["run_method"])()