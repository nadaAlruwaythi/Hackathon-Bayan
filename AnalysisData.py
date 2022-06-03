{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# همة حتى القمة "
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## Resturant"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "- In this analysis I have tried to get in deep understanding of the data and get new insights So my approach is as follows:\n",
    "    * Clean the data as possible"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "### Ceaning Data\n",
    "- Removing Null values\n",
    "- Removing duplicates\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 82,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from gensim import corpora, models, similarities, matutils\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "from sklearn.feature_extraction.text import CountVectorizer\n",
    "\n",
    "import logging\n",
    "logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 83,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div><div id=e50e5098-4484-44b1-b53f-05ad6a1b51d2 style=\"display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;\" onmouseover=\"this.style.backgroundColor='#BA9BF8'\" onmouseout=\"this.style.backgroundColor='#9D6CFF'\" onclick=\"window.commands?.execute('create-mitosheet-from-dataframe-output');\">See Full Dataframe in Mito</div> <script> if (window.commands.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('e50e5098-4484-44b1-b53f-05ad6a1b51d2').style.display = 'flex' </script> <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>address_line1</th>\n",
       "      <th>address_line2</th>\n",
       "      <th>city</th>\n",
       "      <th>food_type</th>\n",
       "      <th>food_type1</th>\n",
       "      <th>location</th>\n",
       "      <th>number_of_reviews</th>\n",
       "      <th>opening_hour</th>\n",
       "      <th>out_of</th>\n",
       "      <th>phone</th>\n",
       "      <th>price_range</th>\n",
       "      <th>restaurant_name</th>\n",
       "      <th>review_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>King Saud Road</td>\n",
       "      <td>Al Khobar 31952 Saudi Arabia</td>\n",
       "      <td>Eastern_Province</td>\n",
       "      <td>, International, Vegan Options, Vegetarian Friendly</td>\n",
       "      <td>International</td>\n",
       "      <td>King Saud Road, Al Khobar 31952 Saudi Arabia</td>\n",
       "      <td>101</td>\n",
       "      <td>+ Add hours</td>\n",
       "      <td>1 of 29 International in Al Khobar</td>\n",
       "      <td>966 13 829 4444</td>\n",
       "      <td>140 -  230</td>\n",
       "      <td>Kempinski Al Othman Hotel</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Prince Musaed Street Dughaither Leisure Island</td>\n",
       "      <td>Al Khobar Saudi Arabia</td>\n",
       "      <td>Eastern_Province</td>\n",
       "      <td>, Seafood, International, Mediterranean</td>\n",
       "      <td>Seafood</td>\n",
       "      <td>Prince Musaed Street Dughaither Leisure Island, Al Khobar Saudi Arabia</td>\n",
       "      <td>916</td>\n",
       "      <td>Closed Now:See all hours</td>\n",
       "      <td>1 of 20 Seafood in Al Khobar</td>\n",
       "      <td>966 3 894 4227</td>\n",
       "      <td>NaN</td>\n",
       "      <td>Al-Sanbok Seafood Restaurant</td>\n",
       "      <td>4.5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Prince Musaid Street</td>\n",
       "      <td>34425 Al Khobar</td>\n",
       "      <td>Eastern_Province</td>\n",
       "      <td>, Japanese, Sushi, Asian</td>\n",
       "      <td>Japanese</td>\n",
       "      <td>Prince Musaid Street, 34425 Al Khobar, Al Khobar Saudi Arabia</td>\n",
       "      <td>59</td>\n",
       "      <td>+ Add hours</td>\n",
       "      <td>1 of 11 Japanese in Al Khobar</td>\n",
       "      <td>971 54 342 9074</td>\n",
       "      <td>160 -  165</td>\n",
       "      <td>Taki Restaurant</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table></div>"
      ],
      "text/plain": [
       "                                    address_line1  \\\n",
       "0                                  King Saud Road   \n",
       "1  Prince Musaed Street Dughaither Leisure Island   \n",
       "2                            Prince Musaid Street   \n",
       "\n",
       "                   address_line2              city  \\\n",
       "0   Al Khobar 31952 Saudi Arabia  Eastern_Province   \n",
       "1         Al Khobar Saudi Arabia  Eastern_Province   \n",
       "2                34425 Al Khobar  Eastern_Province   \n",
       "\n",
       "                                           food_type      food_type1  \\\n",
       "0  , International, Vegan Options, Vegetarian Fri...   International   \n",
       "1            , Seafood, International, Mediterranean         Seafood   \n",
       "2                           , Japanese, Sushi, Asian        Japanese   \n",
       "\n",
       "                                            location number_of_reviews  \\\n",
       "0       King Saud Road, Al Khobar 31952 Saudi Arabia               101   \n",
       "1  Prince Musaed Street Dughaither Leisure Island...               916   \n",
       "2  Prince Musaid Street, 34425 Al Khobar, Al Khob...                59   \n",
       "\n",
       "               opening_hour                              out_of  \\\n",
       "0               + Add hours  1 of 29 International in Al Khobar   \n",
       "1  Closed Now:See all hours        1 of 20 Seafood in Al Khobar   \n",
       "2               + Add hours       1 of 11 Japanese in Al Khobar   \n",
       "\n",
       "             phone  price_range                restaurant_name review_score  \n",
       "0  966 13 829 4444   140 -  230      Kempinski Al Othman Hotel         5.0   \n",
       "1   966 3 894 4227          NaN   Al-Sanbok Seafood Restaurant         4.5   \n",
       "2  971 54 342 9074   160 -  165                Taki Restaurant         5.0   "
      ]
     },
     "execution_count": 83,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df=pd.read_csv('resturantsframe-Clean.csv')\n",
    "df.head(3)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 84,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.\n",
      "Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.\n"
     ]
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 864x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(df['review_score'],bins=9);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 85,
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.\n",
      "Using categorical units to plot a list of strings that are all parsable as floats or dates. If these strings should be plotted as numbers, cast to the appropriate data type before plotting.\n"
     ]
    },
    {
     "data": {
      "image/png": "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qBYRERER8ZNCtYiIiIiInxSqRURERET8pFAtIiIiIuInhWoRERERET8pVIuIiIiI+EmhWkRERETETwrVIiIiIiJ+UqgWEREREfGTQrWIiIiIiJ8UqkVERERE/KRQLSIiIiLiJ4VqERERERE/KVSLiIiIiPhJoVpERERExE8K1SIiIiIiflKoFhERERHxk0K1iIiIiIifFKpFRERERPykUC0iIiIi4ieFahERERERPylUi4iIiIj4SaFaRERERMRPCtUiIiIiIn5SqBYRERER8ZNCtYiIiIiInxSqRURERET8pFAtIiIiIuInhWoRERERET+VaajOz8+nV69eHD58GICnnnqKbt260adPH/r06cPq1asB2LlzJ/369aN79+5MmDABl8tVlsUSEREREbmoyixUb9myhcGDB5OcnGzct23bNmbNmsWSJUtYsmQJXbt2BWD06NE8/fTTrFy5Eq/Xy4IFC8qqWCIiIiIiF12ZheoFCxYwadIkYmNjASgqKuLo0aOMHz+epKQkXn75ZTweD0eOHKG4uJgWLVoA0K9fP1asWFFWxRIRERERuehMXq/XW5Zv0LlzZz744AO8Xi/PP/88kyZNIiwsjAceeIBevXpRv359pk+fzty5cwE4cOAA999/PytXrizLYomIiJ9mzt1MWlYRDWtGkplbTFpWEQAxkUGMGnx1BZdORKR8WcvrjRISEnjttdeM/++44w4WL15M3bp1MZlMxv1er/eU/0srIyMfj6dMzw/OKCYmjLS0vD/dPttjFb3M5VJOrQ+tD322S3N9ZGYW4HK6cTpPjIFxuU7cttmsuJxuMjMLiIoKuWLWx191mculnFofV+76KG9ms4no6NDTP1Zehdi9e/cptc9erxer1UpcXBxpaWnG/enp6UaXERERERGRy0G5hWqv18u0adPIycnB6XQyf/58unbtSnx8PAEBAWzatAmAJUuW0KFDh/IqloiIiIiI38qt+0diYiL3338/gwcPxuVy0a1bN3r16gXAjBkzmDhxIvn5+Vx11VXceeed5VUsERERERG/lXmoXrt2rXF7yJAhDBky5E/PSUxMZOHChWVdFBERERGRMqErKoqIiIiI+EmhWkRERETETwrVIiIiIiJ+UqgWEREREfGTQrWIiIiIiJ8UqkVERERE/KRQLSIiIiLiJ4VqERERERE/KVSLiIiIiPhJoVpERERExE8K1SIiIiIiflKoFhERERHxk0K1iIiIiIifFKpFRERERPykUC0iIiIi4ieFahERERERPylUi4iIiIj4SaFaRERERMRPCtUiIiIiIn5SqBYRERER8ZNCtYiIiIiInxSqRURERET8pFAtIiIiIuInhWoRERERET8pVIuIiIiI+EmhWkRERETETwrVIiIiIiJ+UqgWEREREfGTQrWIiIiIiJ8UqkVERERE/KRQLSIiIiLipzIN1fn5+fTq1YvDhw8DMH/+fHr16kVSUhJPPfUUJSUlALz66qvccMMN9OnThz59+jB79uyyLJaIiIiIyEVlLasX3rJlCxMnTiQ5ORmA/fv38/bbb/PJJ58QEhLCuHHjmDNnDnfffTfbtm3jpZdeomXLlmVVHBERERGRMlNmNdULFixg0qRJxMbGAmC325k0aRKhoaGYTCYaNGjA0aNHAdi2bRtvvfUWSUlJTJkyBYfDUVbFEhERERG56MosVD/77LNcc801xv/x8fFcf/31AGRmZjJ79mxuvPFGCgoKaNSoEaNHj2bRokXk5uby+uuvl1WxREREREQuOpPX6/WW5Rt07tyZDz74gOrVqwOQmprKsGHD6NGjB4888sifnr9jxw7Gjx/P4sWLy7JYIiLip5lzN5OWVUTDmpFk5haTllUEQExkEKMGX13BpRMRKV9l1qf6dPbu3cuwYcO44447GDp0KABHjx5lw4YN9O/fHwCv14vVev7FysjIx+Mp0/ODM4qJCSMtLe9Pt8/2WEUvc7mUU+tD60Of7dJcH5mZBbicbpxOFwAu14nbNpsVl9NNZmYBUVEhV8z6+Ksuc7mUU+vjyl0f5c1sNhEdHXr6x8qrEPn5+dx7772MHDnSCNQAgYGBvPDCCxw6dAiv18vs2bPp2rVreRVLRERERMRv5VZTvXDhQtLT03n33Xd59913gRNdQ0aOHMmUKVN46KGHcDqdXH311dxzzz3lVSwREREREb+Veaheu3YtAHfffTd33333aZ/TvXt3unfvXtZFEREREREpE7qiooiIiIiInxSqRURERET8pFAtIiIiIuInhWoRERERET8pVIuIiIiI+EmhWkRERETETwrVIiIiIiJ+UqgWEREREfGTQrWIiIiIiJ8UqkVERERE/KRQLSIiIiLiJ4VqERERERE/KVSLiIiIiPhJoVpERERExE8K1SIiUmoWy4nDhtlsquCSiIhcWqwVXQAREbk8WCxmFn+7j5wCJzWqhIJytYiIQaFaRERKLSO7mKz8EsKDbRVdFBGRS4q6f4iIiIiI+EmhWkRERETETwrVIiIiIiJ+UqgWEREREfFTqUL1+PHj/3TfiBEjLnphREREREQuR2ed/WPSpEmkpqayadMmMjMzjftdLheHDh0q88KJiIiIiFwOzhqq+/fvz549e9i9ezfdu3c37rdYLLRo0aKsyyYiIiIiclk4a6hu2rQpTZs25brrriMuLq68yiQiIiIiclkp1cVfUlJSGD16NDk5OXi9XuP+pUuXllnBREREREQuF6UK1U8//TT9+vWjcePGmEy6Lq2IiIiIyMlKFaqtViv33HNPWZdFREREROSyVKop9erXr8/u3bvLuiwiIiIiIpelUtVUHzp0iFtvvZVq1aoREBBg3K8+1SIiIiIipQzVo0aNKutyiIiIiIhctkoVqhs0aFDW5RARERERuWyVKlS3adMGk8mE1+s1Zv+IiYnhm2++Oety+fn5DBo0iDfffJPq1auzYcMGnnvuORwOBzfddJNRA75z504mTJhAQUEB11xzDZMnT8ZqLVXRREREREQqXKkGKu7atYudO3eya9cutmzZwrPPPkufPn3OusyWLVsYPHgwycnJABQXFzN+/Hhef/11li9fzrZt2/j6668BGD16NE8//TQrV67E6/WyYMEC/z6ViIiIiEg5KlWoPpndbqdfv36sX7/+rM9bsGABkyZNIjY2FoCtW7dSs2ZNEhISsFqtJCUlsWLFCo4cOUJxcbFx2fN+/fqxYsWK8/8kIiIiIiIVpFR9LLKzs43bXq+Xbdu2kZube9Zlnn322VP+P378ODExMcb/sbGxpKam/un+mJgYUlNTS1MsEREREZFLgsl78nXHzyAxMdHoUw0QHR3NhAkTuPnmm8/5Bp07d+aDDz5g8+bNrFu3jhdeeAGA9evX88477/Dwww/z4osvMmfOHACSk5N58MEHVVstInIJmjl3M2lZRTSsGUlmbvGfbgPERAYxavDVFVxSEZHyVaqa6l27dvn9RnFxcaSlpRn/p6WlERsb+6f709PTjS4j5yMjIx+P55znB2UiJiaMtLS8P90+22MVvczlUk6tD60PfbZLZ31kZhbgcroBcLncuFxunE6X8b/T6cJms+JyusnMLCAqKuQvvT6uhGUul3JqfVy566O8mc0moqNDT/9YaV7A4/Hwn//8hzvuuIPBgwfz6quv4nK5zqsQzZs3Z//+/Rw4cAC3282yZcvo0KED8fHxBAQEsGnTJgCWLFlChw4dzuu1RUTk4rFYzFgsZuO2iIicW6lqql988UV27drFXXfdhcfjYf78+UyfPp3x48eX+o0CAgJ4/vnnefTRR3E4HHTs2JEePXoAMGPGDCZOnEh+fj5XXXUVd95554V9GhER8dvib/eRkV2M1WYhIsTGLe3qVHSRREQueaUK1evWrePjjz/GZrMB0KlTJ3r37l2qUL127Vrjdtu2bfn000//9JzExEQWLlxY2jKLiEgZysgu5nhWodGVQ0REzq1U7Xper9cI1HBiWr2T/xcRERERuZKVKlQnJiYybdo0Dh48yKFDh5g2bZouXS4iIiIi8v+VKlRPmjSJ3NxcBg0axIABA8jKyuJ//ud/yrpsIiIiIiKXhbOG6pKSEsaOHcvGjRt5/vnn2bBhA82aNcNisRAaevrpRERERERErjRnDdUvv/wy+fn5XH31/03iP3XqVHJzc3nllVfKvHAiIiIiIpeDs4bqr776ihdffJHo6GjjvipVqjB9+nTWrFlT5oUTEREREbkcnDVU22w2AgMD/3R/aGgodru9zAolIiIiInI5OWuoNpvN5Ofn/+n+/Pz8876iooiIiIjIX9VZQ3WvXr2YOHEihYWFxn2FhYVMnDiRbt26lXnhREREREQuB2cN1XfddRdhYWFcf/31DBw4kP79+3P99dcTHh7OI488Ul5lFBERERG5pJ31MuVms5mpU6fy4IMPsn37dsxmM82aNSM2Nra8yiciIiIicsk7a6j2iY+PJz4+vqzLIiIiIiJyWSrVFRVFREREROTMFKpFRERERPykUC0iIiIi4ieFahERERERPylUi4iIiIj4SaFaRERERMRPCtUiIiIiIn5SqBYRERER8ZNCtYiIiIiInxSqRURERET8pFAtIiIiIuInhWoRERERET8pVIuIiIiI+EmhWkRERETETwrVIiIiIiJ+UqgWEREREfGTQrWIiIiIiJ8UqkVERERE/KRQLSIiIiLiJ2t5v+FHH33ErFmzjP8PHz5Mnz59KCoqYtOmTQQFBQEwfPhwunbtWt7FExERERE5b+UeqgcMGMCAAQMA2LNnD4888gjDhw/n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TicffPAB6enpJCYmEhUVhcfjYejQoVSvXp3rr7/e+N3abDZWr15NSEgIERERrFixgrvuuougoCDWrFnD448/DsC2bduYOHEidrudLVu20Lt3b2O9pqSkEBERwbp165g9ezZdunQhLS3NOHnwHaDbtGmD1Wrls88+Y/v27bz66qusW7eOgIAA7rzzThwOB5UqVeKXX37hs88+o6SkhJdeeolKlSqxf/9+Ro0axerVq7n11luNmiZfa0J2djbR0dH07t2buLg4nn76acaOHYvNZmPo0KH07t2bqVOnGrXJAEuXLqV///507twZk8nEvffeS2pqKitWrDD2h0OGDDH2Vfn5+cZzfScvWVlZFBQUYLfbyczMJDY2llq1ahEcHEz79u0JCQmhefPmRi34xx9/bOwXDh48CED37t1JSkoywqKvoiIgIICmTZsyatQoo2Zv1apVpwTsqKgonn76aUpKSggLC8NsNrN7924aNWqE1Wplx44dbN26lbCwMAoKCvj444+N2rXg4GAOHjxIUFAQLVu2JDs7m5CQEHbs2EFAQAA33ngj//rXv4yWFJPJxG+//cY111yDzWbj66+/5osvviA6Ohq73c5DDz1knJR3796d+Ph4vvjiCyNMZmVlGcEBYN26dbRq1cr4fZtMJlq0aIHJZCI1NZWcnByuvfZaPvnkE3r37k1mZib16tVjypQpxu/X17KZmZnJzTffzKOPPkpubi5jx46lpKSEgIAAnn/+eZo3b05ISAgBAQHAiRaL1atXGy0lTqfTqCQIDg4mOjqaDz74wKjVbt++Pddee63x/TmdTrp06ULfvn2ZNGkSbdq0wWw207hxY0aMGEFRURFRUVHY7XbCw8MJDg42WnN8+0uHw0FAQICxbo8dO2bs50JCQkhMTDSCaL9+/Zg7d67RGjxlyhQOHTqE1+slNDQUk8nEd999Z7QuhIeHG/tsj8fDbbfdxpgxY6hRowYBAQGEhYUB8N133+HxeOjcuTMbNmwgODiYtLQ0nE4nAQEBXHPNNUYN+LBhw/B4PNjtduP4U7NmTebNm2d8p2PHjsVsNmOxWPj++++Jj48nICCAlStX0rhxY+N7nzJlCg0aNCAgIIClS5ca+8iYmBhMJhMRERHExsbSqVMnzGYzr732Gtdddx1Wq5W0tDRj3xMbG4vX66VRo0aYTCb+93//1zgZWLx4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FuZr5WzSpAkOhwOPx0ODBg2oVasWbrebgoICUlNTCQgIICcnh/3793PgwAEjg/hCcJ06dRg5ciS1a9cmOzubH374AY/Hw8GDBzl+/DiHDx/G7XYTHBxMw4YNMZvNPPHEE/Tp0wc4UeP8wAMPGPvj2bNnU7t2bTweD8eOHTPex+FwcOTIEaMbTVRUFFarleDgYFq3bk1ycjJffvkl2dnZLF++nP3799OyZUvgRLePrKwsMjMzjROLSZMm8dtvv7Fu3TrS0tKoVKkSs2fP5sYbb/xTXouJiSE1NfVP95cXherzdPXVV9OqVSs2bNjAsGHDjJAUERHB9OnTsVqtNG/enEaNGhlNYQDp6elGM9Xy5cuJioqiUaNGFBUVsXfvXqNmIzU1leDgYMLDw6lXr55x1h0VFUV4eDiA0czsq13x2bt3L3Xr1qVmzZrExMRw66234na7yc3NJTExkXr16vH+++/TqFEjYwcUHBzMggULeOedd0hMTDSau2JjY8nIyDCa0ZctW0ZYWBher5cePXoQGBho1IzOnDmT0NBQ0tPTjZrhZcuWERERYRwstm/fbpQzISGBpUuXYrVacbvdfPHFF8TExLB582aqVatGhw4dcLlcHDhwgH/+85906tSJrl27GgfR+++/nxkzZhAcHIzT6SQiIoKDBw+Snp5O06ZNjabV5cuXExoaahy04US/1lmzZhln3pUqVWLEiBFGDXFsbCxff/01HTt2ZMiQITRp0oQdO3bQrFkzmjVrRvPmzQEYNWoU7du3ByA5OZn777+fG264geXLlzN16lTefvtt6tWrR926dalRowZpaWkcO3bM2MF16NCBMWPGEBwcTFJSEm+99RYWi4UtW7awbds2IiIi6Nq1q9HH94YbbuDrr7+mXr16PPDAA4wZM8bYufpCdU5ODo8//jidO3emSpUq3HXXXRw/fhybzcYzzzzDunXr6Natm/G9x8TEAFCrVi2WLVuGw+GgVatWxMbGEh8fj8PhIDQ0lMTERGJjYzl27Bh79uxhxowZVK9enYkTJxIcHMzevXu58847jf5uI0eOJC4uDq/Xy/z586levTq9evXCZrMxb948Vq9ezffff2/UYPu6QDRv3pysrCzj5KhTp05UqlSJnJwc+vXrh8fjoW7dugwePJg333yTu+++m6uuuopjx47RunVrrrrqKrZv307NmjV58sknMZlMPPTQQ/z4449069aN22+/nbS0NOrUqYPT6aRHjx4cPnyYXbt28cADD/D444+Tl5fH6NGjmTx5MlFRUeTk5AAYoatnz54sW7YMt9tNt27d+Oc//4nJZDIOjh6Ph8mTJ9OkSRPj9zpw4EDeeOMNQkNDjQOlyWQiNjaW2267zejLeu2115KQkEBJSYlRgzV+/HhatWpFTk4OH330Eddeey2bNm2ioKCAoKAgY7vetWsX6enpjBkzhrCwMBo2bMj3339v9DNesWIFHTp04Oeff6ZNmzaEhYXxxRdf0KhRo1PGUvhER0czcuRIvvvuOzIzM6lbty7Lli1jxowZLFu2jHfffZdDhw4ZB3bfMqNHj+a1115jxIgRNGvWjPT0dL7++mtat25N3759cTgcOBwOOnToQGpqKrGxsXz11VfGdn733XdjsVhYuHAhY8aMoVOnTgDGwbV58+bGibWvFvno0aPGuI3GjRvjcrm4/fbbGTp0KMuWLcNisbB48WJuv/12I5j/85//NPq4P/bYY4SHh2O1Wo0ubwDPPfccHTp04MiRI7jdbqOZ2m63GycsZrOZjz76iNtvv93oO/r9999TuXJlHA4HMTEx2O12Dh8+TPv27Y0uUwkJCZhMJrp06ULTpk2BE2EiPDycvXv3EhQURElJCUeOHDFaH33b4Pr16xkyZAj16tXDZrPx73//m7fffhubzcbf//5347ler5eoqCjq1KlDVFQUe/bsoaCgAJfLxeOPP878+fOx2+18+eWXvPDCCwwfPpy6devStWtXCgoK2LVrF9deey0TJ0401r2vz/LQoUPp16+f0fria5G97rrrACgsLKRu3br89ttvVKpUiWXLlhEUFERubi61a9fmpptuMrrH+cYQ+bqp2Ww2CgoKjCBuNpspKCjghRdeML4js9nMM888Q79+/SgoKAAwTiYKCgqoW7cuhw8fBk5UhPlaCRs2bGgc+xo3bky9evWMvvJRUVGMGjWKxYsXk5WVZfx+69ati9vtprCwkJ49ezJixAjsdjspKSl8+OGHHDhwgNGjR9OzZ09sNhtjx45lyJAhRgvAyJEjje/k5ZdfZseOHdhsNho3bmx0o2jVqhUjR440toWIiAjjeDdkyBCioqJwOBx88803xnZw00038dlnnwFQtWpVoqOjKSgowOFwMGjQIEJCQvB4PHTp0oX/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G2jqG3RotVqNPv0rV640Qj5gjF3xddG89957T9nHxsTEEB8fT2FhIVarlX//+988/vjjmM1mMjIyCAgI4JFHHgH+b0CjT3p6urG9VQSF6vOUkpLCm2++yb59+9ixYwcLFy4kLy+Phg0bMn36dDweD4cPHyYlJYXmzZuzf/9+1q1bR69evThy5AgWi4UXX3yRZcuWUbduXVJSUoymGJvNZvxo8vLyaNKkCV6vl++++45x48bx7rvvctVVV+F2u2nTpg0mk4lHH33UKFtSUhKVK1c2+jM+9NBDxln85s2b2bx5M5MnTzb6gfoGNviaplq0aMFTTz0FQFxcHJs2bSI0NJSioiIcDocxcOw///kPw4YNA04MSHz22WcJCQnBbDZTqVIlAgICuOGGG3jnnXf4+eefcblcfPbZZ8bB1tf06guyo0aNoqioiFtuuYWPP/6YkpISYmNjCQgIYMeOHUybNo3c3FwqV64MwKOPPspzzz3HgAEDaN++PSaTiVq1alGpUiW++OIL8vPziY6O5j//+Q9hYWFERERQt25dPvnkEzIyMmjfvj2ZmZkEBQXx+++/8/TTT1OpUiWji8OPP/7I8OHDOX78ODfccAMRERG0atWKAwcOUFhYiNvtxul08tFHH2E2myksLCQ8PJxPPvmEo0eP0r9/fxwOB1WrVmX9+vU8+eSThISEsHjxYg4cOIDT6SQoKIhhw4YRExPD+vXr+e9//0toaCg1a9YkISGBbdu20aRJE9LT0/F4PMTExJCXl8ftt99Oz549AYwBItHR0YwYMQKz2UzPnj35+eef2b59O5UrV2b+/PnUqVOH2267jVmzZtG0aVOio6OZPHky9erVM7o0VKpUiR07dtC/f38aNGhAy5Ytsdls3HTTTaxbt87YGb700kusWbOGfv36MXToUKPrTGZmptEs63Q6WbduHQ6Hg+DgYAYMGMDSpUtxOBwcO3aM6dOn8+OPP5Kamkp2djb169cnPz/f6Cvq8XgYM2YM2dnZtG7dmrZt21JcXIzdbmfQoEFMnz6devXqcfz4ceLj45k8eTLLli1j1apVhIaG0qpVK1577TUABg0axKuvvsr7779PYmIiq1evNmqhCgoKjFok31iHtLQ00tPTad68OXv37sVqteLxeIwBOr7BbU6nk7p16/Lcc88REhJC27ZtjQOAr5XhwIEDPPvsszz1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qc5OntrS91vyjTPIzs4mNDTUeO2T11tERAQNGzbku+++M37LvkGNQUFBRusMYAwq89Wi+17Dt52aTCaCgoKMygY4MRi8sLDQeB2r1WpM+VZcXHzK56levTpFRUWkp6fj9XqNVkdfv92wsDDsdjvZ2dlkZ2cbZfGVxzcjxjXXXMOxY8c4dOgQjRs3Zvfu3SxdupTNmzczYcIEo2xdu3Y1uoZ99913RujyzYDh6ypRrVo1I2QdOnTICNO+wey+PsYnr1uHw0FISIhxf1BQEDVr1iQjI8OoqbdarUawslqt5OXlndL07vV6qVu3LjabzRg3kp2dbbRG+MYm+bof+L4/u91O69at+eabb4iLizMGL1ssllOOib5jqe9zms1mgoODjX7GvuPhyft5X4j3PTcxMZFff/2V5s2bExoayrfffmuc5ADG8cZXWeWbxca3/ftOvHzrzHccNplMRk1spUqVyMrKMrZ/37R5vsHxJ2+fvmO8b7B9bGwshw8fxuv1EhgYSExMDAcPHiQ+Pp4jR44Yxxff+vb1Ya5Xrx4//PCD0QUDMPqJT5gwgUOHDlG7dm1CQkLIzc01ZuZyOBxUq1aNBx98kMmTJ1NSUmL8llesWMHcuXN5//33KS4upkqVKtSsWZPff//daDm94YYb6NKlC++99x7Hjx83piwNCwsjLy8Pu93O3/72N4YPH86oUaOMKSCvv/56Zs+efUo/7Pr16/Pjjz8ax07fzFYOh4OUlBQCAwNp164dd999N5MnTyY/P5+rrrqK5557zpjxpLwpVIuIiMgF81U4Va9enfvuu48RI0bQtGlT1qxZw7/+9S9jIO7kyZONEwzAOIlbtGiRMci+T58+hISEEBoayj333MOHH374p7C0detWJk2aRFFREQ0bNuTZZ58lNDSUAwcOMHbsWPLy8qhSpQrPPfecMauESHlQqBYREZGL4t1336Vdu3bGrEDlvbxIRVKfahEREbkofF0PKmp5kYqkmmoRERERET+pplpERERExE8K1SIiIiIiflKoFhERERHxk0K1iMgloHPnzvz6668V8t6TJk2ic+fOzJw5s0zfZ+7cufz73/8u0/cQEako1oougIiIVKz58+fz1VdfERcXV6bvM3jw4DJ9fRGRiqRQLSJSSt9//z0zZ84kISGBPXv24HK5mDx5Mh999BH169fn3nvvBWDcuHHG/507d6ZXr15899135OTkMGzYMDZv3sz27duxWq288cYbxgUq5syZw65duygpKeGee+6hf//+AKxdu5Y33njDuArl2LFjadmyJa+88gq//PILx48fp2HDhsyYMeOMZd+zZw9TpkwhOzsbk8nE0KFDueWWW7j99tvxer3cd999TJo06ZRLcP/xsz/77LMEBwdTUFDAxx9/zLfffvuncjVr1ozOnTvz2muv0aRJEwAee+wx/va3v5GRkUFWVhZPP/00qampTJkyhZSUFJxOJz179uTBBx/k4Ycf5oYbbmDAgAH8/PPPDBo0iDVr1pCQkMDrr79OQUEB/fr1Y8KECcZVJ/v378+QIUMu5lctInLeFKpFRM6D72pujRo14p133mHmzJlUr179rMs4HA4WLFjA8uXLeeKJJ1i0aBGJiYk88sgjLFq0iAcffBCAgIAAFi1aRGpqKn379qV58+bYbDZmzpzJBx98QGRkJHv27OGee+5h1apVABw5coRly5YZl1k/HZfLxUMPPcSYMWPo1q0bqampDBgwgJo1azJnzhwaNmzI+++/T1RU1Fk/x549e1izZg3x8fEkJyefsVy33norn3zyCU2aNCEnJ4eNGzcydepU3nvvPeO1Ro8ezd13303nzp1xOBzcd9991KhRg27durF27VoGDBjAunXriImJYcOGDdx2222sXbuWp59+mrfffpvOnTtz//33k5aWxrRp0xg8eLBxuWcRkYqgUC0ich6qVatGo0aNAGjcuDGLFi06Z6ju1q0bAAkJCVSuXJnExEQAatSoQU5OjvG8QYMGAVClShWuv/56Nm7ciMVi4fjx49x9993G80wmEwcPHgSgRYsWZw3UAMnJyTgcDqMcVapUoVu3bqxbt46WLVuW+rNXrVqV+Ph4ANavX3/Gct16663079+fcePGsWzZMjp37nzK5akLCwv58ccfycnJ4V//+pdx365du7j33nt57rnncLlcfPvttzz00EOsX7+eTp06kZmZSdOmTcnIyGDs2LFs3bqVtm3bMnHiRAVqEalwCtUiIuchMDDQuG0ymfB6vcZfH6fTecoydrvduG2z2c742icHQ4/Hg9Vqxe1207ZtW/73f//XeCwlJYXY2FhWr15NcHDwOcvsdrsxmUyn3Of1enG5XOdc9mQnv5fH4zljuSwWC40bN+arr77ik08+Yfz48ae8jsfjwev1Mm/ePIKCggDIzMwkICCAkJAQGjVqxJdffkl+fj59+vTh9ddfZ82aNXTp0gWTycQNN9zAypUr2bBhAxs3buS1117jk08+KfM+4SIiZ6NTexERP0VGRrJt2zYAUlNT+eGHHy7odRYtWgTA0aNH2bhxI23btqVt27asX7+evXv3AvD111/Tu3dviouLS/26derUwWq1Gl1GUlNTWblyJdddd90FlRM4Z7kGDhzIf/7zH4qKimjVqtUpy4aGhtKiRQveffddAHJzcxk8eDBffPEFAF27duWll16ibdu2hIaGUqtWLf7zn/8YNe1PPPEEy5cvp2fPnkyaNInQ0FCj5l5EpKKoplpExE933HEHTz75JN27d6d69eq0adPmgl7H4XDQt29fnE4nEydOpHbt2gBMmTKFxx9/HK/XawxuDAkJKfXr2mw2Xn/9dZ555hleeeUV3G43jzzyyAWXE6BevXpnLVfnzp2ZPHky991332mXnzFjBlOnTiUpKYmSkhJ69epF7969AejSpQtTp07lySefBKBdu3bMnj2bq6++GoCHH36YCRMmMH/+fCwWC126dOHaa6+94M8iInIxmLwnt1mKiIiIiMh5U021iMhfwKeffsrbb7992seSkpIYNmzYOV/jscceY//+/ad9bObMmdSpU8evMoqI/JWpplpERERExE8aqCgiIiIi4ieFahERERERPylUi4iIiIj4SaFaRERERMRPCtUiIiIiIn76f/j+6hQo/xbIAAAAAElFTkSuQmCC\n",
      "text/plain": [
       "<Figure size 864x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.histplot(df['number_of_reviews'],bins=9);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 86,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "1830"
      ]
     },
     "execution_count": 86,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 87,
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "RangeIndex: 1830 entries, 0 to 1829\n",
      "Data columns (total 13 columns):\n",
      " #   Column             Non-Null Count  Dtype \n",
      "---  ------             --------------  ----- \n",
      " 0   address_line1      1830 non-null   object\n",
      " 1   address_line2      1677 non-null   object\n",
      " 2   city               1830 non-null   object\n",
      " 3   food_type          1809 non-null   object\n",
      " 4   food_type1         1766 non-null   object\n",
      " 5   location           1830 non-null   object\n",
      " 6   number_of_reviews  1830 non-null   object\n",
      " 7   opening_hour       1830 non-null   object\n",
      " 8   out_of             1639 non-null   object\n",
      " 9   phone              1500 non-null   object\n",
      " 10  price_range        476 non-null    object\n",
      " 11  restaurant_name    1830 non-null   object\n",
      " 12  review_score       1830 non-null   object\n",
      "dtypes: object(13)\n",
      "memory usage: 186.0+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "address_line1           0\n",
       "address_line2         153\n",
       "city                    0\n",
       "food_type              21\n",
       "food_type1             64\n",
       "location                0\n",
       "number_of_reviews       0\n",
       "opening_hour            0\n",
       "out_of                191\n",
       "phone                 330\n",
       "price_range          1354\n",
       "restaurant_name         0\n",
       "review_score            0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isna().sum()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.dropna(axis=0, how=\"any\", thresh=None, subset=None, inplace=True)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "334"
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "<class 'pandas.core.frame.DataFrame'>\n",
      "Int64Index: 334 entries, 0 to 1778\n",
      "Data columns (total 13 columns):\n",
      " #   Column             Non-Null Count  Dtype \n",
      "---  ------             --------------  ----- \n",
      " 0   address_line1      334 non-null    object\n",
      " 1   address_line2      334 non-null    object\n",
      " 2   city               334 non-null    object\n",
      " 3   food_type          334 non-null    object\n",
      " 4   food_type1         334 non-null    object\n",
      " 5   location           334 non-null    object\n",
      " 6   number_of_reviews  334 non-null    object\n",
      " 7   opening_hour       334 non-null    object\n",
      " 8   out_of             334 non-null    object\n",
      " 9   phone              334 non-null    object\n",
      " 10  price_range        334 non-null    object\n",
      " 11  restaurant_name    334 non-null    object\n",
      " 12  review_score       334 non-null    object\n",
      "dtypes: object(13)\n",
      "memory usage: 36.5+ KB\n"
     ]
    }
   ],
   "source": [
    "df.info()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Removing duplicates"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 51,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0"
      ]
     },
     "execution_count": 51,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.duplicated().sum()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 52,
   "metadata": {},
   "outputs": [],
   "source": [
    "df.drop_duplicates(inplace=True)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "334"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "len(df)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Riyadh              129\n",
       "Eastern_Province    116\n",
       "Jeddah               89\n",
       "Name: city, dtype: int64"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['city'].value_counts()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "#### Show City"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x432 with 1 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "sns.set(style='darkgrid')\n",
    "plt.rcParams['figure.figsize'] = 6,6\n",
    "sorted_counts = df['city'].value_counts()\n",
    "plt.pie(sorted_counts, startangle = 90, counterclock = False, autopct='%.0f%%' );\n",
    "plt.title('City');\n",
    "plt.legend(sorted_counts.index);"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div><div id=b5b76306-5b82-4c51-a5f8-410e3b832966 style=\"display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;\" onmouseover=\"this.style.backgroundColor='#BA9BF8'\" onmouseout=\"this.style.backgroundColor='#9D6CFF'\" onclick=\"window.commands?.execute('create-mitosheet-from-dataframe-output');\">See Full Dataframe in Mito</div> <script> if (window.commands.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('b5b76306-5b82-4c51-a5f8-410e3b832966').style.display = 'flex' </script> <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>address_line1</th>\n",
       "      <th>address_line2</th>\n",
       "      <th>city</th>\n",
       "      <th>food_type</th>\n",
       "      <th>food_type1</th>\n",
       "      <th>location</th>\n",
       "      <th>number_of_reviews</th>\n",
       "      <th>opening_hour</th>\n",
       "      <th>out_of</th>\n",
       "      <th>phone</th>\n",
       "      <th>price_range</th>\n",
       "      <th>restaurant_name</th>\n",
       "      <th>review_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>count</th>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "      <td>334</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>unique</th>\n",
       "      <td>315</td>\n",
       "      <td>226</td>\n",
       "      <td>3</td>\n",
       "      <td>245</td>\n",
       "      <td>43</td>\n",
       "      <td>327</td>\n",
       "      <td>91</td>\n",
       "      <td>41</td>\n",
       "      <td>334</td>\n",
       "      <td>306</td>\n",
       "      <td>258</td>\n",
       "      <td>326</td>\n",
       "      <td>7</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>top</th>\n",
       "      <td>King Saud Road</td>\n",
       "      <td>Riyadh Saudi Arabia</td>\n",
       "      <td>Riyadh</td>\n",
       "      <td>, Cafe</td>\n",
       "      <td>Italian</td>\n",
       "      <td>King Saud Road, Al Khobar 31952 Saudi Arabia</td>\n",
       "      <td>1</td>\n",
       "      <td>+ Add hours</td>\n",
       "      <td>35 of 40 American in Al Khobar</td>\n",
       "      <td>966 9200 02690</td>\n",
       "      <td>50 -  200</td>\n",
       "      <td>Lusin</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>freq</th>\n",
       "      <td>5</td>\n",
       "      <td>25</td>\n",
       "      <td>129</td>\n",
       "      <td>13</td>\n",
       "      <td>45</td>\n",
       "      <td>3</td>\n",
       "      <td>42</td>\n",
       "      <td>139</td>\n",
       "      <td>1</td>\n",
       "      <td>5</td>\n",
       "      <td>6</td>\n",
       "      <td>3</td>\n",
       "      <td>136</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table></div>"
      ],
      "text/plain": [
       "         address_line1         address_line2    city food_type food_type1  \\\n",
       "count              334                   334     334       334        334   \n",
       "unique             315                   226       3       245         43   \n",
       "top     King Saud Road   Riyadh Saudi Arabia  Riyadh    , Cafe    Italian   \n",
       "freq                 5                    25     129        13         45   \n",
       "\n",
       "                                            location number_of_reviews  \\\n",
       "count                                            334               334   \n",
       "unique                                           327                91   \n",
       "top     King Saud Road, Al Khobar 31952 Saudi Arabia                 1   \n",
       "freq                                               3                42   \n",
       "\n",
       "       opening_hour                          out_of           phone  \\\n",
       "count           334                             334             334   \n",
       "unique           41                             334             306   \n",
       "top     + Add hours  35 of 40 American in Al Khobar  966 9200 02690   \n",
       "freq            139                               1               5   \n",
       "\n",
       "       price_range restaurant_name review_score  \n",
       "count          334             334          334  \n",
       "unique         258             326            7  \n",
       "top      50 -  200           Lusin         4.0   \n",
       "freq             6               3          136  "
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.describe()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['address_line1', 'address_line2', 'city', 'food_type', 'food_type1',\n",
       "       'location', 'number_of_reviews', 'opening_hour', 'out_of', 'phone',\n",
       "       'price_range', 'restaurant_name', 'review_score'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div><div id=6624b590-fb4e-4715-a51d-4912182894a7 style=\"display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;\" onmouseover=\"this.style.backgroundColor='#BA9BF8'\" onmouseout=\"this.style.backgroundColor='#9D6CFF'\" onclick=\"window.commands?.execute('create-mitosheet-from-dataframe-output');\">See Full Dataframe in Mito</div> <script> if (window.commands.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('6624b590-fb4e-4715-a51d-4912182894a7').style.display = 'flex' </script> <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>address_line1</th>\n",
       "      <th>address_line2</th>\n",
       "      <th>city</th>\n",
       "      <th>food_type</th>\n",
       "      <th>food_type1</th>\n",
       "      <th>location</th>\n",
       "      <th>number_of_reviews</th>\n",
       "      <th>opening_hour</th>\n",
       "      <th>out_of</th>\n",
       "      <th>phone</th>\n",
       "      <th>price_range</th>\n",
       "      <th>restaurant_name</th>\n",
       "      <th>review_score</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1778</th>\n",
       "      <td>3878 Prince Muhammad Bin Abdulaziz Rd</td>\n",
       "      <td>As Sulimaniyah</td>\n",
       "      <td>Riyadh</td>\n",
       "      <td>, Italian, American, International</td>\n",
       "      <td>Italian</td>\n",
       "      <td>3878 Prince Muhammad Bin Abdulaziz Rd, As Sulimaniyah,, Riyadh 12242 Saudi Arabia</td>\n",
       "      <td>2</td>\n",
       "      <td>Open Now:See all hours</td>\n",
       "      <td>97 of 97 Italian in Riyadh</td>\n",
       "      <td>966 59 336 5566</td>\n",
       "      <td>11 -  139</td>\n",
       "      <td>Cookery restaurant</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table></div>"
      ],
      "text/plain": [
       "                              address_line1    address_line2    city  \\\n",
       "1778  3878 Prince Muhammad Bin Abdulaziz Rd   As Sulimaniyah  Riyadh   \n",
       "\n",
       "                                 food_type food_type1  \\\n",
       "1778    , Italian, American, International    Italian   \n",
       "\n",
       "                                               location number_of_reviews  \\\n",
       "1778  3878 Prince Muhammad Bin Abdulaziz Rd, As Suli...                 2   \n",
       "\n",
       "                opening_hour                      out_of            phone  \\\n",
       "1778  Open Now:See all hours  97 of 97 Italian in Riyadh  966 59 336 5566   \n",
       "\n",
       "     price_range      restaurant_name review_score  \n",
       "1778   11 -  139   Cookery restaurant         5.0   "
      ]
     },
     "execution_count": 45,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.tail(1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 48,
   "metadata": {
    "scrolled": true
   },
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div><div id=e95895eb-d4df-4564-87fb-cbd717c37378 style=\"display:none; background-color:#9D6CFF; color:white; width:200px; height:30px; padding-left:5px; border-radius:4px; flex-direction:row; justify-content:space-around; align-items:center;\" onmouseover=\"this.style.backgroundColor='#BA9BF8'\" onmouseout=\"this.style.backgroundColor='#9D6CFF'\" onclick=\"window.commands?.execute('create-mitosheet-from-dataframe-output');\">See Full Dataframe in Mito</div> <script> if (window.commands.hasCommand('create-mitosheet-from-dataframe-output')) document.getElementById('e95895eb-d4df-4564-87fb-cbd717c37378').style.display = 'flex' </script> <table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>address_line1</th>\n",
       "      <th>location</th>\n",
       "      <th>city</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>240</th>\n",
       "      <td>Al Khalij Rd</td>\n",
       "      <td>Al Khalij Rd, Al Mazrooa, Al Hofuf 36362, Al Hofuf, Al Ahsa 31982 Saudi Arabia</td>\n",
       "      <td>Eastern_Province</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1545</th>\n",
       "      <td>4100 Northern Ring Rd</td>\n",
       "      <td>4100 Northern Ring Rd, Riyadh 13311 Saudi Arabia</td>\n",
       "      <td>Riyadh</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1053</th>\n",
       "      <td>Sultana Center</td>\n",
       "      <td>Sultana Center, Opposite Danube Tahlia, Jeddah 23326 Saudi Arabia</td>\n",
       "      <td>Jeddah</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1105</th>\n",
       "      <td>jeddah</td>\n",
       "      <td>jeddah, prince sultan road, Al zahra district Opposite of Salamah Center, Jeddah 21411 Saudi Arabia</td>\n",
       "      <td>Jeddah</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1630</th>\n",
       "      <td>2649 Northern Ring Branch Road An Nakheel</td>\n",
       "      <td>2649 Northern Ring Branch Road An Nakheel, Riyadh 12385 13512 Saudi Arabia</td>\n",
       "      <td>Riyadh</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1044</th>\n",
       "      <td>Ice Land</td>\n",
       "      <td>Ice Land, Prince Sultan Street Al-Shallal Theme Park Branch, Jeddah 21465 Saudi Arabia</td>\n",
       "      <td>Jeddah</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>421</th>\n",
       "      <td>Al Aziziyah</td>\n",
       "      <td>Al Aziziyah, Dammam Saudi Arabia</td>\n",
       "      <td>Eastern_Province</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1079</th>\n",
       "      <td>King Abdullah Rd Abraj Al Hilal - Tower 3</td>\n",
       "      <td>King Abdullah Rd Abraj Al Hilal - Tower 3, Jeddah 2973 Saudi Arabia</td>\n",
       "      <td>Jeddah</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>404</th>\n",
       "      <td>Al Khaleej Road</td>\n",
       "      <td>Al Khaleej Road, Al Qatif 31911 Saudi Arabia</td>\n",
       "      <td>Eastern_Province</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1088</th>\n",
       "      <td>AL Rawda</td>\n",
       "      <td>AL Rawda, Qasim Zainah, Jeddah 23414 Saudi Arabia</td>\n",
       "      <td>Jeddah</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table></div>"
      ],
      "text/plain": [
       "                                  address_line1  \\\n",
       "240                                Al Khalij Rd   \n",
       "1545                      4100 Northern Ring Rd   \n",
       "1053                             Sultana Center   \n",
       "1105                                     jeddah   \n",
       "1630  2649 Northern Ring Branch Road An Nakheel   \n",
       "1044                                   Ice Land   \n",
       "421                                 Al Aziziyah   \n",
       "1079  King Abdullah Rd Abraj Al Hilal - Tower 3   \n",
       "404                             Al Khaleej Road   \n",
       "1088                                   AL Rawda   \n",
       "\n",
       "                                               location              city  \n",
       "240   Al Khalij Rd, Al Mazrooa, Al Hofuf 36362, Al H...  Eastern_Province  \n",
       "1545   4100 Northern Ring Rd, Riyadh 13311 Saudi Arabia            Riyadh  \n",
       "1053  Sultana Center, Opposite Danube Tahlia, Jeddah...            Jeddah  \n",
       "1105  jeddah, prince sultan road, Al zahra district ...            Jeddah  \n",
       "1630  2649 Northern Ring Branch Road An Nakheel, Riy...            Riyadh  \n",
       "1044  Ice Land, Prince Sultan Street Al-Shallal Them...            Jeddah  \n",
       "421                    Al Aziziyah, Dammam Saudi Arabia  Eastern_Province  \n",
       "1079  King Abdullah Rd Abraj Al Hilal - Tower 3, Jed...            Jeddah  \n",
       "404        Al Khaleej Road, Al Qatif 31911 Saudi Arabia  Eastern_Province  \n",
       "1088  AL Rawda, Qasim Zainah, Jeddah 23414 Saudi Arabia            Jeddah  "
      ]
     },
     "execution_count": 48,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.loc[:,['address_line1','location','city']].sample(10,random_state=1)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 58,
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "data": {
      "text/plain": [
       " Italian           45\n",
       " American          37\n",
       " International     30\n",
       " Cafe              28\n",
       " Indian            28\n",
       " Lebanese          20\n",
       " Chinese           17\n",
       " French            14\n",
       " Middle Eastern    13\n",
       " Mediterranean     10\n",
       " Seafood           10\n",
       " Japanese          10\n",
       " Steakhouse         7\n",
       " Fast Food          7\n",
       " Asian              6\n",
       "Restaurants         6\n",
       " European           4\n",
       " Turkish            3\n",
       " Arabic             3\n",
       " Grill              3\n",
       " Pizza              3\n",
       " Bar                3\n",
       " Mexican            3\n",
       " Barbecue           3\n",
       " Portuguese         2\n",
       " Halal              2\n",
       " Greek              1\n",
       " British            1\n",
       " Armenian           1\n",
       " Egyptian           1\n",
       " Street Food        1\n",
       " Southwestern       1\n",
       " Moroccan           1\n",
       " Belgian            1\n",
       " South American     1\n",
       " Pakistani          1\n",
       " Dutch              1\n",
       " Latin              1\n",
       " Swedish            1\n",
       " African            1\n",
       " Contemporary       1\n",
       " Hawaiian           1\n",
       " Caribbean          1\n",
       "Name: food_type1, dtype: int64"
      ]
     },
     "execution_count": 58,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df['food_type1'].value_counts()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Index(['address_line1', 'address_line2', 'city', 'food_type', 'food_type1',\n",
       "       'location', 'number_of_reviews', 'opening_hour', 'out_of', 'phone',\n",
       "       'price_range', 'restaurant_name', 'review_score'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.columns"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
