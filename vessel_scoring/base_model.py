import vessel_scoring.add_measures

class BaseModel(object):
    # def train_on_messages(self, messages):
    #     messages = AddMeasures(messages, self.windows)        
    #     y_train = utils.is_fishy(train_data)
    #     model.fit(train_data, y_train)
    #     return model

    def predict_messages(self, messages):
        for msg in vessel_scoring.add_measures.AddMeasures(messages, self.windows):
            if (msg.get('timestamp', None) is not None and
                msg.get('speed', None) is not None and
                msg.get('course', None) is not None):
                msg['measure_new_score'] = float(self.predict_proba({
                            key: [value]
                            for key, value in msg.iteritems()
                            })[0][1])
            yield msg
        
    def dump_arg_dict(self):
        return None

    def dump_dict(self):
        args = self.dump_arg_dict()
        if args is None:
            return None
        model_class = type(self)
        return {'model': "%s.%s" % (model_class.__module__, model_class.__name__),
                'args': args,
                }

