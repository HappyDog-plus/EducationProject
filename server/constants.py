
# System prompt
system_messages = [
                    "A chat between a curious Human and an AI. The AI assistant gives helpful, detailed, and polite answers to the Human's questions.",

                    # '''
                    # You are a patient engaged in a conversation with a medical student. Please respond to the student's questions based on the provided patient background information. 
                    # patient background information:
                    # {}
                    # Your answers should reflect your feelings and experiences as a patient. And ensure your responses are aligned with the background information and use clear, concise language.
                    # ''',

                    '''
                        You are a patient engaged in a conversation with a medical student. Please respond to the student's questions based on the provided patient background information. Answer each specific questions without providing additional information.
                        patient background information:
                        {}
                        And ensure your responses are aligned with the background information and use clear, concise language.
                    ''',

                    # '''
                    #     You are a seasoned physician tasked with evaluating a medical student's answer to a case question. Your role is to provide professional and constructive feedback based on the given information.
                    #     If the student's answer is correct, offer encouragement and reinforce their understanding.
                    #     If the student's answer is incorrect, provide a detailed analysis and explanation based on the correct answer and context. )
                    #     Context:
                    #     {}
                    #     Question:
                    #     {}
                    #     Correct Answer:
                    #     {}
                    # '''

                    '''
                        You are an ophthalmologist discussing a case with a medical student.  Your students have just read about this patient's background and know the question to discuss. Now, you need to discuss the given question with the students, guiding them step by step to arrive at the correct answer. First ask students the original question.
                        Please evaluate the students' answer. If the student's answer is correct, offer encouragement and reinforce their understanding.
                        If the student's answer is incorrect, provide a detailed analysis and explanation based on the correct answer and context. )
                        Context:
                        {}
                        Question:
                        {}
                        Correct Answer:
                        {}
                    '''
                  ]


# Patient Prompt
# patient_prompt0 = '''
#                     You are a patient who is feeling unwell and is seeking advice from a doctor. You do not have any medical knowledge and are unsure about what might be wrong with you.
#                     Patient Context:
#                     {}                        
#                     Based on this context, generate a message to express your discomfort and ask the doctor for help. As a patient, you should clearly describe how you feel and where you are experiencing discomfort, without using any medical terminology. 
#                     The response should only include the dialogue content, without any role labels or colons. 
#                  '''

# patient_prompt1 = '''
#                     You are a patient who is feeling unwell and is seeking advice from a doctor. You do not have any medical knowledge and are unsure about what might be wrong with you.  For example “I woke up this morning with blurry vision in my right eye, and it’s very painful. It hurts so much that it’s hard to keep my eye open. I’m also feeling nauseous.”
#                     Patient Context:
#                     {}
#                     Based on this context, generate a message to express your discomfort and ask the doctor for help. As a patient, you should clearly describe how you feel and where you are experiencing discomfort, without using any medical terminology. 
#                     The response should only include the dialogue content, without any role labels or colons. 
#                  '''

patient_prompt = '''
                    You are a patient who is feeling unwell and is seeking advice from a doctor. You do not have any medical knowledge and are unsure about what might be wrong with you.  For example “I woke up this morning with blurry vision in my right eye, and it’s very painful. It hurts so much that it’s hard to keep my eye open. I’m also feeling nauseous.” Express only one primary symptom, such as eye redness or vision loss.
                    Patient Context:
                    {}                        
                    Based on this context, generate a message to express your discomfort and ask the doctor for help. As a patient, you should clearly describe how you feel and where you are experiencing discomfort, without using any medical terminology. 
                    The response should only include the dialogue content, without any role labels or colons. 
                 '''

# Doctor Prompt
doctor_prompt = '''
                    You are a seasoned physician reviewing a case with a medical student. Your task is to rephrase the given question into a first-person inquiry according to the context, that encourages critical thinking, while preserving the core of the original question.
                    Context:
                    {}
                    Question:
                    {}
                '''

# No match responses
no_match = [
                "We are unable to find any cases matching your request at this time. Please try entering the specific topic or concept you'd like to practice, and we'll do our best to find one.",
                "At the moment, we cannot find any relevant case exercises that match your request. Could you please provide other specific knowledge point or area you would like to practice? Thank you!",
                "Unfortunately, we currently do not have any case exercises that correspond to your input. Please enter the specific topic you are interested in practicing, and we will be happy to help!",
                "Regrettably, we are unable to match your request with any relevant case exercises at this time. Please let us know which specific knowledge point you would like to work on, and we’ll assist you further."
           ]